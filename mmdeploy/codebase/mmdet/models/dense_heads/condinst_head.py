# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch
from mmdet.models.utils import aligned_bilinear
from mmengine.config import ConfigDict
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops.nms import multiclass_nms


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.CondInstBboxHead.predict_by_feat')
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
        featmap_sizes, device=device, with_stride=True)
    all_level_points = [i[:, :2] for i in all_level_points_strides]
    all_level_strides = [i[:, 2] for i in all_level_points_strides]

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                              self.cls_out_channels)
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

    bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    scores = flatten_cls_scores
    score_factors = flatten_score_factors
    param_preds = flatten_param_preds
    scores = scores * score_factors

    # get post processing config
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

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
    points = points.unsqueeze(0).repeat(batch_size, 1, 1)
    strides = strides.unsqueeze(0).repeat(batch_size, 1)
    param_preds = param_preds[batch_inds, inds, :]
    points = points[batch_inds, inds, :]
    strides = strides[batch_inds, inds]
    results = dict(
        dets=dets,
        labels=labels,
        param_preds=param_preds,
        points=points,
        strides=strides)
    return results


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.CondInstMaskHead.forward')
def condinst_mask_head__forward(self, x: tuple,
                                positive_infos: Dict[str, torch.Tensor]):
    mask_feats = self.mask_feature_head(x)

    param_preds = positive_infos['param_preds']
    points = positive_infos['points']
    strides = positive_infos['strides']

    batch_size = points.shape[0]
    num_insts = points.shape[1]
    hw = mask_feats.size()[-2:]
    mask_feats = mask_feats.unsqueeze(1).repeat(1, num_insts, 1, 1, 1)

    points = points.reshape(-1, 1, 2).unsqueeze(0)
    locations = self.prior_generator.single_level_grid_priors(
        hw, level_idx=0, device=mask_feats.device)
    locations = locations.unsqueeze(0).repeat(batch_size, 1,
                                              1).reshape(batch_size, 1, -1, 2)
    centers = points.reshape(batch_size, -1, 1, 2)
    rel_coordinates = (centers - locations).permute(0, 1, 3, 2).float()
    rel_coordinates /= (strides[:, :, None, None] * self.size_of_interest)
    rel_coords = rel_coordinates.reshape(batch_size, -1, 2, hw[0], hw[1])
    mask_head_inputs = torch.cat([rel_coords, mask_feats], dim=2)

    weights, biases = _parse_dynamic_params(self, param_preds)
    mask_preds = _dynamic_conv_forward(mask_head_inputs, weights, biases)
    mask_preds = mask_preds.reshape(batch_size, num_insts, hw[0], hw[1])
    mask_preds = aligned_bilinear(
        mask_preds, int(self.mask_feat_stride / self.mask_out_stride))
    return (mask_preds, )


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.CondInstMaskHead.predict_by_feat')
def condinst_mask_head__predict_by_feat(self,
                                        mask_preds: Tensor,
                                        results_list: Dict[str, torch.Tensor],
                                        batch_img_metas: List[dict],
                                        rescale: bool = True,
                                        **kwargs):
    cfg = self.test_cfg

    dets = results_list['dets']
    labels = results_list['labels']
    img_hw = batch_img_metas[0]['img_shape'][:2]

    mask_preds = mask_preds.sigmoid()
    mask_preds = aligned_bilinear(mask_preds, self.mask_out_stride)
    mask_preds = mask_preds[:, :, :img_hw[0], :img_hw[1]]
    masks = (mask_preds > cfg.mask_thr).float()

    return dets, labels, masks


def _parse_dynamic_params(self, params: Tensor):
    """parse the dynamic params for dynamic conv."""
    batch_size = params.shape[0]
    num_insts = params.shape[1]
    params = params.permute(1, 0, 2)
    params_splits = list(
        torch.split_with_sizes(
            params, self.weight_nums + self.bias_nums, dim=2))

    weight_splits = params_splits[:self.num_layers]
    bias_splits = params_splits[self.num_layers:]

    for idx in range(self.num_layers):
        if idx < self.num_layers - 1:
            weight_splits[idx] = weight_splits[idx].reshape(
                batch_size, num_insts, self.in_channels, -1)
        else:
            weight_splits[idx] = weight_splits[idx].reshape(
                batch_size, num_insts, 1, -1)

    return weight_splits, bias_splits


def _dynamic_conv_forward(features: Tensor, weights: List[Tensor],
                          biases: List[Tensor]):
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
