# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from mmdet.models.layers import mask_matrix_nms
from torch import Tensor
from torch.nn import functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.SOLOHead.predict_by_feat')
def solohead__predict_by_feat__default(ctx, self,
                                       mlvl_mask_preds: List[Tensor],
                                       mlvl_cls_scores: List[Tensor],
                                       batch_img_metas: List[Dict], **kwargs):
    """Rewrite `predict_by_feat` of `SOLOHead` for default backend."""

    def empty_results(cls_scores, ori_shape):
        batch_size = cls_scores.size(1)
        scores = cls_scores.new_ones(batch_size, 0, 1)
        masks = cls_scores.new_zeros(batch_size, 0, *ori_shape)
        labels = cls_scores.new_ones(batch_size, 0)
        bboxes = cls_scores.new_zeros(batch_size, 0, 4)
        dets = torch.cat([bboxes, scores], -1)
        return dets, labels, masks

    batch_size = mlvl_cls_scores[0].size(0)
    cfg = self.test_cfg
    mlvl_cls_scores = [
        item.permute(0, 2, 3, 1).view(batch_size, -1, self.cls_out_channels)
        for item in mlvl_cls_scores
    ]
    assert len(mlvl_mask_preds) == len(mlvl_cls_scores)
    batch_mlvl_cls_scores = torch.cat(mlvl_cls_scores, dim=1)
    batch_mlvl_mask_preds = torch.cat(mlvl_mask_preds, dim=1)
    featmap_size = batch_mlvl_mask_preds.size()[-2:]
    h, w = batch_img_metas[0]['img_shape'][:2]
    upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)
    cls_labels = batch_mlvl_cls_scores.argmax(dim=-1)
    batch_mlvl_cls_scores = torch.max(batch_mlvl_cls_scores, -1).values
    score_mask = (batch_mlvl_cls_scores > cfg.score_thr)
    batch_mlvl_cls_scores = batch_mlvl_cls_scores.where(
        score_mask, batch_mlvl_cls_scores.new_zeros(1)).view(-1)
    if len(batch_mlvl_cls_scores) == 0:
        return empty_results(batch_mlvl_cls_scores,
                             batch_img_metas[0]['ori_shape'][:2])
    cls_labels = cls_labels.view(-1)

    # Filter the mask mask with an area is smaller than
    # stride of corresponding feature level
    batch_lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
    strides = batch_mlvl_cls_scores.new_ones(batch_lvl_interval[-1])
    strides[:batch_lvl_interval[0]] *= self.strides[0]
    for lvl in range(1, self.num_levels):
        strides[batch_lvl_interval[lvl -
                                   1]:batch_lvl_interval[lvl]] *= self.strides[
                                       lvl]

    mask_preds = batch_mlvl_mask_preds.view(-1, featmap_size[0],
                                            featmap_size[1])
    masks = (mask_preds > cfg.mask_thr).int()
    sum_masks = masks.sum((1, 2))
    cls_scores = batch_mlvl_cls_scores
    keep = sum_masks > strides
    if keep.sum() == 0:
        return empty_results(batch_mlvl_cls_scores,
                             batch_img_metas[0]['ori_shape'][:2])
    cls_scores = batch_mlvl_cls_scores.where(
        keep, batch_mlvl_cls_scores.new_zeros(1))
    sum_masks = sum_masks.where(keep, sum_masks.new_ones(1))

    # maskness.
    mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
    cls_scores *= mask_scores
    sum_masks = sum_masks.where(keep, sum_masks.new_zeros(1))

    scores, labels, _, keep_inds = mask_matrix_nms(
        masks,
        cls_labels,
        cls_scores,
        mask_area=sum_masks,
        nms_pre=cfg.nms_pre,
        max_num=cfg.max_per_img,
        kernel=cfg.kernel,
        sigma=cfg.sigma,
        filter_thr=cfg.filter_thr)
    # mask_matrix_nms may return an empty Tensor

    if len(keep_inds) == 0:
        return empty_results(cls_scores, batch_img_metas[0]['ori_shape'][:2])
    mask_preds = mask_preds[keep_inds]
    mask_preds = F.interpolate(
        mask_preds.unsqueeze(0), size=upsampled_size,
        mode='bilinear')[:, :, :h, :w]
    mask_preds = F.interpolate(
        mask_preds, size=batch_img_metas[0]['ori_shape'][:2], mode='bilinear')
    labels = labels.reshape(batch_size, -1)
    bboxes = scores.new_zeros(scores.shape[-1], 4).view(batch_size, -1, 4)

    dets = torch.cat([bboxes, scores.reshape(batch_size, -1, 1)], dim=-1)
    return dets, labels, mask_preds
