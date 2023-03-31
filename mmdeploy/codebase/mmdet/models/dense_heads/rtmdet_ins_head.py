# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from packaging import version
from torch import Tensor

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops import ONNXNMSop, TRTBatchedNMSop


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.rtmdet_ins_head.'
    'RTMDetInsHead.predict_by_feat')
def rtmdet_ins_head__predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        kernel_preds: List[Tensor],
        mask_feat: Tensor,
        score_factors: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True):
    """Rewrite `predict_by_feat` of `RTMDet-Ins` for default backend.
    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.
    Args:
        ctx: Context that contains original meta information.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.
    Returns:
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """
    assert len(cls_scores) == len(bbox_preds)
    device = cls_scores[0].device
    cfg = self.test_cfg if cfg is None else cfg
    batch_size = bbox_preds[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, device=device, with_stride=True)

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                              self.cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_kernel_preds = [
        kernel_pred.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                self.num_gen_params)
        for kernel_pred in kernel_preds
    ]
    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_kernel_preds = torch.cat(flatten_kernel_preds, dim=1)
    priors = torch.cat(mlvl_priors)
    tl_x = (priors[..., 0] - flatten_bbox_preds[..., 0])
    tl_y = (priors[..., 1] - flatten_bbox_preds[..., 1])
    br_x = (priors[..., 0] + flatten_bbox_preds[..., 2])
    br_y = (priors[..., 1] + flatten_bbox_preds[..., 3])
    bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    # directly multiply score factor and feed to nms
    max_scores, _ = torch.max(flatten_cls_scores, 1)
    mask = max_scores >= cfg.score_thr
    scores = flatten_cls_scores.where(mask, flatten_cls_scores.new_zeros(1))

    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    mask_thr_binary = cfg.get('mask_thr_binary', 0.5)

    return _nms_with_mask_static(self, priors, bboxes, scores,
                                 flatten_kernel_preds, mask_feat,
                                 max_output_boxes_per_class, iou_threshold,
                                 score_threshold, pre_top_k, keep_top_k,
                                 mask_thr_binary)


def _nms_with_mask_static(self,
                          priors: Tensor,
                          boxes: Tensor,
                          scores: Tensor,
                          kernels: Tensor,
                          mask_feats: Tensor,
                          max_output_boxes_per_class: int = 1000,
                          iou_threshold: float = 0.5,
                          score_threshold: float = 0.05,
                          pre_top_k: int = -1,
                          keep_top_k: int = -1,
                          mask_thr_binary: float = 0.5):
    """Wrapper for `multiclass_nms` with ONNXRuntime.
    Args:
        ctx (ContextCaller): The context with additional information.
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    if version.parse(torch.__version__) < version.parse('1.13.0'):
        max_output_boxes_per_class = torch.LongTensor(
            [max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)

    # pre topk
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.squeeze(0).topk(pre_top_k)
        boxes = boxes[:, topk_inds, :]
        scores = scores[:, topk_inds, :]
        kernels = kernels[:, topk_inds, :]
        priors = priors[topk_inds, :]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSop.apply(boxes, scores,
                                       max_output_boxes_per_class,
                                       iou_threshold, score_threshold)

    cls_inds = selected_indices[:, 1]
    box_inds = selected_indices[:, 2]

    scores = scores[:, cls_inds, box_inds].unsqueeze(2)
    boxes = boxes[:, box_inds, ...]
    kernels = kernels[:, box_inds, :]
    priors = priors[box_inds, :]
    dets = torch.cat([boxes, scores], dim=2)
    labels = cls_inds.unsqueeze(0)

    # pad
    dets = torch.cat((dets, dets.new_zeros((1, 1, 5))), 1)
    labels = torch.cat((labels, labels.new_zeros((1, 1))), 1)
    kernels = torch.cat((kernels, kernels.new_zeros(1, 1, kernels.shape[2])),
                        1)
    priors = torch.cat((priors, priors.new_zeros(1, 4)), 0)

    # topk or sort
    is_use_topk = keep_top_k > 0 and \
        (torch.onnx.is_in_onnx_export() or keep_top_k < dets.shape[1])
    if is_use_topk:
        _, topk_inds = dets[:, :, -1].topk(keep_top_k, dim=1)
    else:
        _, topk_inds = dets[:, :, -1].sort(dim=1, descending=True)
    topk_inds = topk_inds.squeeze(0)
    dets = dets[:, topk_inds, ...]
    labels = labels[:, topk_inds, ...]
    kernels = kernels[:, topk_inds, ...]
    priors = priors[topk_inds, ...]
    mask_logits = _mask_predict_by_feat_single(self, mask_feats, kernels[0],
                                               priors)
    stride = self.prior_generator.strides[0][0]
    mask_logits = F.interpolate(
        mask_logits.unsqueeze(0), scale_factor=stride, mode='bilinear')
    masks = mask_logits.sigmoid()
    return dets, labels, masks


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.models.'
    'dense_heads.rtmdet_ins_head._nms_with_mask_static',
    backend='tensorrt')
def _nms_with_mask_static__tensorrt(self,
                                    priors: Tensor,
                                    boxes: Tensor,
                                    scores: Tensor,
                                    kernels: Tensor,
                                    mask_feats: Tensor,
                                    max_output_boxes_per_class: int = 1000,
                                    iou_threshold: float = 0.5,
                                    score_threshold: float = 0.05,
                                    pre_top_k: int = -1,
                                    keep_top_k: int = -1,
                                    mask_thr_binary: float = 0.5):
    """Wrapper for `multiclass_nms` with TensorRT.
    Args:
        ctx (ContextCaller): The context with additional information.
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    keep_top_k = max_output_boxes_per_class if keep_top_k < 0 else min(
        max_output_boxes_per_class, keep_top_k)
    dets, labels, inds = TRTBatchedNMSop.apply(boxes, scores,
                                               int(scores.shape[-1]),
                                               pre_top_k, keep_top_k,
                                               iou_threshold, score_threshold,
                                               -1, True)
    # inds shape: (batch, n_boxes)
    # retain shape info
    batch_size = boxes.size(0)

    dets_shape = dets.shape
    label_shape = labels.shape
    dets = dets.reshape([batch_size, *dets_shape[1:]])
    labels = labels.reshape([batch_size, *label_shape[1:]])
    kernels = kernels[:, inds.reshape(-1), ...]
    priors = priors[inds.reshape(-1), ...]
    mask_logits = _mask_predict_by_feat_single(self, mask_feats, kernels[0],
                                               priors)
    stride = self.prior_generator.strides[0][0]
    mask_logits = F.interpolate(
        mask_logits.unsqueeze(0), scale_factor=stride, mode='bilinear')
    masks = mask_logits.sigmoid()
    return dets, labels, masks


def _mask_predict_by_feat_single(self, mask_feat, kernels, priors):
    """decode mask with dynamic conv."""
    num_inst = priors.shape[0]
    h, w = mask_feat.size()[-2:]
    if num_inst < 1:
        return torch.empty(
            size=(num_inst, h, w),
            dtype=mask_feat.dtype,
            device=mask_feat.device)
    if len(mask_feat.shape) < 4:
        mask_feat.unsqueeze(0)
    coord = self.prior_generator.single_level_grid_priors(
        (h, w), level_idx=0).reshape(1, -1, 2).to(mask_feat.device)
    num_inst = priors.shape[0]
    points = priors[:, :2].reshape(-1, 1, 2)
    strides = priors[:, 2:].reshape(-1, 1, 2)
    relative_coord = (points - coord).permute(0, 2, 1) / (
        strides[..., 0].reshape(-1, 1, 1) * 8)
    relative_coord = relative_coord.reshape(num_inst, 2, h, w)

    mask_feat = torch.cat(
        [relative_coord, mask_feat.repeat(num_inst, 1, 1, 1)], dim=1)
    weights, biases = _parse_dynamic_params(self, kernels)

    n_layers = len(weights)
    x = mask_feat.flatten(2)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        # replace dynamic conv with bmm
        x = torch.bmm(weight, x)
        x = x + bias[:, :, None]
        if i < n_layers - 1:
            x = x.clamp_(min=0)
    x = x.reshape(num_inst, h, w)
    return x


def _parse_dynamic_params(self, flatten_kernels):
    """split kernel head prediction to conv weight and bias."""
    n_inst = flatten_kernels.size(0)
    n_layers = len(self.weight_nums)
    params_splits = list(
        torch.split_with_sizes(
            flatten_kernels, self.weight_nums + self.bias_nums, dim=1))
    weight_splits = params_splits[:n_layers]
    bias_splits = params_splits[n_layers:]
    for idx in range(n_layers):
        if idx < n_layers - 1:
            weight_splits[idx] = weight_splits[idx].reshape(
                n_inst, self.dyconv_channels, -1)
        else:
            weight_splits[idx] = weight_splits[idx].reshape(n_inst, 1, -1)
    return weight_splits, bias_splits
