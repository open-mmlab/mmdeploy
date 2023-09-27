# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional

import torch
from mmdet.models.utils import aligned_bilinear, relative_coordinate_maps
from mmdet.utils import InstanceList
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops.nms import multiclass_nms


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

    all_level_points_strides = self.prior_generator.grid_priors(
        featmap_sizes, device=device, with_stride=True
    )
    all_level_points = [priors[:, :2] for priors in all_level_points_strides]
    all_level_strides = [priors[:, 2] for priors in all_level_points_strides]

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

    dets, labels, inds = multiclass_nms(
        bboxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k,
        output_index=True,
    )

    batch_inds = torch.arange(batch_size, device=bboxes.device).view(-1, 1)
    param_preds = param_preds.repeat(batch_size, 1, 1)
    points = points.unsqueeze(0).repeat(batch_size, 1, 1)
    strides = strides.unsqueeze(0).repeat(batch_size, 1)
    param_preds = param_preds.index_select(0, batch_inds[0]).index_select(1, inds[0])
    points = points.index_select(0, batch_inds[0]).index_select(1, inds[0])
    strides = strides.index_select(0, batch_inds[0]).index_select(1, inds[0])

    results_list = []
    for dets_, labels_, param_preds_, points_, strides_ in zip(
        dets, labels, param_preds, points, strides
    ):
        results = InstanceData()
        results.dets = dets_
        results.bboxes = dets_[:, :4]
        results.scores = dets_[:, 4]
        results.labels = labels_
        results.param_preds = param_preds_
        results.points = points_
        results.strides = strides_
        results_list.append(results)
    return results_list


@FUNCTION_REWRITER.register_rewriter(
    "mmdet.models.dense_heads.CondInstMaskHead.forward"
)
def condinst_mask_head__forward(self, x: tuple, positive_infos: InstanceList):
    mask_feats = self.mask_feature_head(x)

    param_preds = [positive_info.get("param_preds") for positive_info in positive_infos]
    points = [positive_info.get("points") for positive_info in positive_infos]
    strides = [positive_info.get("strides") for positive_info in positive_infos]
    param_preds = torch.stack(param_preds, dim=0)
    points = torch.stack(points, dim=0)
    strides = torch.stack(strides, dim=0)

    batch_size = points.shape[0]
    num_insts = points.shape[1]
    hw = mask_feats.size()[-2:]

    points = points.reshape(-1, 1, 2).unsqueeze(0)
    coord = self.prior_generator.single_level_grid_priors(
        hw, level_idx=0, device=mask_feats.device
    )
    coord = coord.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    relative_coord = (points - coord).permute(0, 1, 3, 2) / (
        strides[:, :, None, None] * self.size_of_interest
    )
    relative_coord = relative_coord.reshape(batch_size, num_insts, 2, hw[0], hw[1])

    mask_feats = torch.cat(
        (relative_coord, mask_feats.unsqueeze(1).repeat(1, num_insts, 1, 1, 1)), dim=2
    )

    weights, biases = _parse_dynamic_params(self, param_preds)
    mask_preds = _dynamic_conv_forward(self, mask_feats, weights, biases, num_insts)
    mask_preds = mask_preds.reshape(batch_size, num_insts, hw[0], hw[1])
    mask_preds = [
        aligned_bilinear(
            mask_pred.unsqueeze(0),
            int(self.mask_feat_stride / self.mask_out_stride),
        ).squeeze(0)
        for mask_pred in mask_preds
    ]

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
    cfg = self.test_cfg

    dets = [results.dets.unsqueeze(0) for results in results_list]
    labels = [results.labels.unsqueeze(0) for results in results_list]
    img_hw = [img_meta["img_shape"][:2] for img_meta in batch_img_metas]

    mask_preds = [mask_pred.sigmoid().unsqueeze(0) for mask_pred in mask_preds]
    mask_preds = [
        aligned_bilinear(mask_pred, self.mask_out_stride) for mask_pred in mask_preds
    ]
    mask_preds = [
        mask_preds[i][:, :, : img_hw[i][0], : img_hw[i][1]]
        for i in range(len(mask_preds))
    ]

    masks = [mask_pred > cfg.mask_thr for mask_pred in mask_preds]
    masks = [mask.float() for mask in masks]

    return dets, labels, masks


def _parse_dynamic_params(self, params: Tensor):
    """parse the dynamic params for dynamic conv."""
    batch_size = params.shape[0]
    num_insts = params.shape[1]
    params = params.permute(1, 0, 2)
    params_splits = list(
        torch.split_with_sizes(params, self.weight_nums + self.bias_nums, dim=2)
    )

    weight_splits = params_splits[: self.num_layers]
    bias_splits = params_splits[self.num_layers :]

    for idx in range(self.num_layers):
        if idx < self.num_layers - 1:
            weight_splits[idx] = weight_splits[idx].reshape(
                batch_size, num_insts, self.in_channels, -1
            )
        else:
            weight_splits[idx] = weight_splits[idx].reshape(
                batch_size, num_insts, 1, -1
            )

    return weight_splits, bias_splits


def _dynamic_conv_forward(
    self, features: Tensor, weights: List[Tensor], biases: List[Tensor], num_insts: int
):
    """dynamic forward, each layer follow a relu."""
    n_layers = len(weights)
    x = features.flatten(0, 1).flatten(2)
    for i, (w, b) in enumerate(zip(weights, biases)):
        # replace dynamic conv with bmm
        w = w.flatten(0, 1)
        b = b.flatten(0, 1).unsqueeze(2)
        x = torch.bmm(w, x)
        x = x + b
        if i < n_layers - 1:
            x = x.clamp_(min=0)
    return x
