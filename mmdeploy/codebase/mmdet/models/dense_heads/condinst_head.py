# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
from mmdet.models.utils import aligned_bilinear, relative_coordinate_maps
from mmdet.models.utils.misc import empty_instances
from mmdet.utils import InstanceList
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from packaging import version
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops import ONNXNMSop, multiclass_nms


@FUNCTION_REWRITER.register_rewriter(
    "mmdet.models.dense_heads.CondInstBboxHead.predict_by_feat"
)
def condinst_bbox_head__predict_by_feat(
    self,
    cls_scores: List[Tensor],
    bbox_preds: List[Tensor],
    score_factors: Optional[List[Tensor]] = None,
    param_preds: Optional[List[Tensor]] = None,
    batch_img_metas: Optional[List[dict]] = None,
    cfg: Optional[ConfigDict] = None,
    rescale: bool = False,
    with_nms: bool = True,
):
    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg

    assert len(cls_scores) == len(bbox_preds)
    device = bbox_preds[0].device
    cfg = self.test_cfg if cfg is None else cfg
    batch_size = bbox_preds[0].shape[0]
    featmap_sizes = [cls_score.shape[-2:] for cls_score in cls_scores]
    pre_topk = cfg.get("nms_pre", -1)

    all_level_points_strides = self.prior_generator.grid_priors(
        featmap_sizes, device=device, with_stride=True
    )
    all_level_points = [i[:, :2] for i in all_level_points_strides]
    all_level_strides = [i[:, 2] for i in all_level_points_strides]

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_score_factors = [
        score_factor.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
        for score_factor in score_factors
    ]
    flatten_param_preds = [
        param_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_params)
        for param_pred in param_preds
    ]
    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_score_factors = torch.cat(flatten_score_factors, dim=1).sigmoid()
    flatten_param_preds = torch.cat(flatten_param_preds, dim=1)

    points = torch.cat(all_level_points)
    strides = torch.cat(all_level_strides)
    tl_x = points[..., 0] - flatten_bbox_preds[..., 0]
    tl_y = points[..., 1] - flatten_bbox_preds[..., 1]
    br_x = points[..., 0] + flatten_bbox_preds[..., 2]
    br_y = points[..., 1] + flatten_bbox_preds[..., 3]
    #
    bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)  # decode
    scores = flatten_cls_scores
    score_factors = flatten_score_factors
    param_preds = flatten_param_preds
    scores = scores * score_factors

    # get post processing config
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get("iou_threshold", post_params.iou_threshold)
    score_threshold = cfg.get("score_thr", post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get("max_per_img", post_params.keep_top_k)
    nms_type = cfg.nms.get("type")

    # Rewrite nms for onnx convertion
    if with_nms:
        (
            batched_dets,
            batched_labels,
            batched_param_preds,
            batched_points,
            batched_strides,
        ) = _nms_condinst(
            self,
            bboxes,
            scores,
            score_factors,
            param_preds,
            points,
            strides,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            pre_top_k,
            keep_top_k,
            nms_type,
        )

    results_list = []
    for dets, labels, param_preds, points, strides in zip(
        batched_dets,
        batched_labels,
        batched_param_preds,
        batched_points,
        batched_strides,
    ):
        results = InstanceData()
        results.dets = dets
        results.bboxes = dets[:, :4]
        results.scores = dets[:, -1]
        results.labels = labels
        results.param_preds = param_preds
        results.points = points
        results.strides = strides
        results_list.append(results)
    return results_list


def _nms_condinst(
    self,
    bboxes: Tensor,
    scores: Tensor,
    score_factors: Tensor,
    param_preds: Tensor,
    points: Tensor,
    strides: Tensor,
    max_output_boxes_per_class: int = 1000,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    nms_type: str = "nms",
    output_index: bool = False,
):
    """Wrapper for `multiclass_nms` with ONNXRuntime."""
    if version.parse(torch.__version__) < version.parse("1.13.0"):
        max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = bboxes.shape[0]

    # pre topk
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.squeeze(0).topk(pre_top_k)
        batch_inds = torch.arange(batch_size, device=bboxes.device).view(-1, 1).long()
        bboxes = bboxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]
        score_factors = score_factors[batch_inds, topk_inds, :]
        param_preds = param_preds[batch_inds, topk_inds, :]
        points = points[topk_inds, :]
        strides = strides[topk_inds]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSop.apply(
        bboxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )

    batch_inds = selected_indices[:, 0]
    cls_inds = selected_indices[:, 1]
    box_inds = selected_indices[:, 2]

    # index by nms output
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)  # pre: 2
    bboxes = bboxes[batch_inds, box_inds, ...]
    param_preds = param_preds[batch_inds, box_inds, ...]
    points = points[box_inds, :]
    strides = strides[box_inds]
    dets = torch.cat((bboxes, scores), dim=1)  # pre: 2

    # batch all
    batched_dets = dets.repeat(batch_size, 1, 1)
    batch_template = torch.arange(
        0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device
    )
    batched_dets = batched_dets.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_dets.new_zeros(1),
    )

    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where(
        (batch_inds == batch_template.unsqueeze(1)), batched_labels.new_ones(1) * -1
    )

    batched_param_preds = param_preds.repeat(batch_size, 1, 1)
    batched_param_preds = batched_param_preds.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_param_preds.new_zeros(1),
    )

    batched_points = points.unsqueeze(0).repeat(batch_size, 1, 1)
    batched_points = batched_points.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_points.new_zeros(1),
    )

    batched_strides = strides.unsqueeze(0).repeat(batch_size, 1)
    batched_strides = batched_strides.where(
        (batch_inds == batch_template.unsqueeze(1)), batched_strides.new_zeros(1)
    )

    N = batched_dets.shape[0]
    # expand tensor to eliminate [0, ...] tensor
    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
    batched_labels = torch.cat((batched_labels, batched_labels.new_zeros((N, 1))), 1)
    batched_param_preds = torch.cat(
        (
            batched_param_preds,
            batched_param_preds.new_zeros(N, 1, batched_param_preds.shape[2]),
        ),
        1,
    )
    batched_points = torch.cat((batched_points, batched_points.new_zeros((N, 1, 2))), 1)
    batched_strides = torch.cat((batched_strides, batched_strides.new_zeros((N, 1))), 1)

    # sort
    is_use_topk = keep_top_k > 0 and (
        torch.onnx.is_in_onnx_export() or keep_top_k < batched_dets.shape[1]
    )
    if is_use_topk:
        _, topk_inds = batched_dets[:, :, -1].topk(keep_top_k, dim=1)
    else:
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(
        batch_size, dtype=topk_inds.dtype, device=topk_inds.device
    ).view(-1, 1)

    batched_dets = batched_dets.index_select(0, topk_batch_inds[0]).index_select(1, topk_inds[0])
    batched_labels = batched_labels.index_select(0, topk_batch_inds[0]).index_select(1, topk_inds[0])
    batched_param_preds = batched_param_preds.index_select(0, topk_batch_inds[0]).index_select(1, topk_inds[0])
    batched_points = batched_points.index_select(0, topk_batch_inds[0]).index_select(1, topk_inds[0])
    batched_strides = batched_strides.index_select(0, topk_batch_inds[0]).index_select(1, topk_inds[0])
    # slice and recover the tensor
    return (
        batched_dets,
        batched_labels,
        batched_param_preds,
        batched_points,
        batched_strides,
    )


@FUNCTION_REWRITER.register_rewriter(
    "mmdet.models.dense_heads.CondInstMaskHead.forward_single"
)
def condinst_mask_head__forward_single(
    self, mask_feat: Tensor, positive_info: InstanceData
):
    """Rewrite function about forwarding features of a each image."""
    pos_param_preds = positive_info.get("param_preds")
    pos_points = positive_info.get("points")
    pos_strides = positive_info.get("strides")

    num_inst = pos_param_preds.shape[0]
    mask_feat = mask_feat[None].repeat(num_inst, 1, 1, 1)
    _, _, H, W = mask_feat.size()
    if num_inst == 0:
        return (pos_param_preds.new_zeros((0, 1, H, W)),)

    locations = self.prior_generator.single_level_grid_priors(
        mask_feat.size()[2:], 0, device=mask_feat.device
    )

    rel_coords = relative_coordinate_maps(
        locations, pos_points, pos_strides, self.size_of_interest, mask_feat.size()[2:]
    )
    mask_head_inputs = torch.cat([rel_coords, mask_feat], dim=1)
    # mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

    weights, biases = _parse_dynamic_params(self, pos_param_preds)
    mask_preds = _dynamic_conv_forward(
        self, mask_head_inputs, weights, biases, num_inst
    )
    mask_preds = mask_preds.reshape(-1, H, W)
    mask_preds = aligned_bilinear(
        mask_preds.unsqueeze(0), int(self.mask_feat_stride / self.mask_out_stride)
    ).squeeze(0)

    return (mask_preds,)


@FUNCTION_REWRITER.register_rewriter(
    "mmdet.models.dense_heads.CondInstMaskHead.predict_by_feat"
)
def condinst_mask_head__predict_by_feat(
    self,
    mask_preds: List[Tensor],
    results_list: InstanceList,
    batch_img_metas: List[dict],
    rescale: bool = True,
    **kwargs
):
    assert len(mask_preds) == len(results_list) == len(batch_img_metas)

    for img_id in range(len(batch_img_metas)):
        img_meta = batch_img_metas[img_id]
        results = results_list[img_id]
        bboxes = results.bboxes
        mask_pred = mask_preds[img_id]
        if bboxes.shape[0] == 0 or mask_pred.shape[0] == 0:
            results_list[img_id] = empty_instances(
                [img_meta], bboxes.device, task_type="mask", instance_results=[results]
            )[0]
        else:
            im_mask = self._predict_by_feat_single(
                mask_preds=mask_pred, bboxes=bboxes, img_meta=img_meta, rescale=rescale
            )
            results.masks = im_mask.float()
    return results_list


def _parse_dynamic_params(self, params: Tensor):
    """parse the dynamic params for dynamic conv."""
    num_insts = params.size(0)
    params_splits = list(
        torch.split_with_sizes(params, self.weight_nums + self.bias_nums, dim=1)
    )
    weight_splits = params_splits[: self.num_layers]
    bias_splits = params_splits[self.num_layers :]
    for idx in range(self.num_layers):
        if idx < self.num_layers - 1:
            weight_splits[idx] = weight_splits[idx].reshape(
                num_insts, self.in_channels, -1
            )
        else:
            weight_splits[idx] = weight_splits[idx].reshape(num_insts, 1, -1)
    return weight_splits, bias_splits


def _dynamic_conv_forward(
    self, features: Tensor, weights: List[Tensor], biases: List[Tensor], num_insts: int
):
    """dynamic forward, each layer follow a relu."""
    n_layers = len(weights)
    x = features.flatten(2)
    for i, (w, b) in enumerate(zip(weights, biases)):
        # replace dynamic conv with bmm
        x = torch.bmm(w, x)
        x = x + b[:, :, None]
        if i < n_layers - 1:
            x = x.clamp_(min=0)  # relu
    return x
