# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn.functional as F
from mmengine.config import ConfigDict
from torch import Tensor

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops.nms import multiclass_nms


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
    scores = flatten_cls_scores

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
                          bboxes: Tensor,
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
        bboxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
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
    dets, labels, inds = multiclass_nms(
        bboxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k,
        output_index=True)

    batch_size = bboxes.shape[0]
    batch_inds = torch.arange(batch_size, device=bboxes.device).view(-1, 1)
    kernels = kernels[batch_inds, inds, :]
    priors = priors.unsqueeze(0).repeat(batch_size, 1, 1)
    priors = priors[batch_inds, inds, :]
    mask_logits = _mask_predict_by_feat_single(self, mask_feats, kernels,
                                               priors)
    stride = self.prior_generator.strides[0][0]
    mask_logits = F.interpolate(
        mask_logits, scale_factor=stride, mode='bilinear')
    masks = mask_logits.sigmoid()
    return dets, labels, masks


def _mask_predict_by_feat_single(self, mask_feat, kernels, priors):
    """decode mask with dynamic conv."""
    num_inst = priors.shape[1]
    batch_size = priors.shape[0]
    hw = mask_feat.size()[-2:]
    coord = self.prior_generator.single_level_grid_priors(
        hw, level_idx=0).to(mask_feat.device)
    coord = coord.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    priors = priors.unsqueeze(2)
    points = priors[..., :2]
    relative_coord = (points - coord).permute(0, 1, 3, 2) / (
        priors[..., 2:3] * 8)
    relative_coord = relative_coord.reshape(batch_size, num_inst, 2, hw[0],
                                            hw[1])

    mask_feat = torch.cat(
        [relative_coord,
         mask_feat.unsqueeze(1).repeat(1, num_inst, 1, 1, 1)],
        dim=2)
    weights, biases = _parse_dynamic_params(self, kernels)

    n_layers = len(weights)
    x = mask_feat.flatten(0, 1).flatten(2)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        # replace dynamic conv with bmm
        weight = weight.flatten(0, 1)
        bias = bias.flatten(0, 1).unsqueeze(2)
        x = torch.bmm(weight, x)
        x = x + bias
        if i < n_layers - 1:
            x = x.clamp_(min=0)
    x = x.reshape(batch_size, num_inst, hw[0], hw[1])
    return x


def _parse_dynamic_params(self, flatten_kernels):
    """split kernel head prediction to conv weight and bias."""
    batch_size = flatten_kernels.shape[0]
    n_inst = flatten_kernels.shape[1]
    n_layers = len(self.weight_nums)
    params_splits = list(
        torch.split_with_sizes(
            flatten_kernels, self.weight_nums + self.bias_nums, dim=2))
    weight_splits = params_splits[:n_layers]
    bias_splits = params_splits[n_layers:]
    for idx in range(n_layers):
        channel = self.dyconv_channels if idx < n_layers - 1 else 1
        weight_splits[idx] = weight_splits[idx].reshape(
            batch_size, n_inst, channel, -1)

    return weight_splits, bias_splits
