# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
from mmdet.models.layers import mask_matrix_nms
from mmdet.utils import OptConfigType
from torch import Tensor
from torch.nn import functional as F

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.solo_head.'
    'SOLOHead.predict_by_feat')
def solohead__predict_by_feat(self,
                              mlvl_mask_preds: List[Tensor],
                              mlvl_cls_scores: List[Tensor],
                              batch_img_metas: List[Dict],
                              cfg: OptConfigType = None,
                              **kwargs):
    """Rewrite `predict_by_feat` of `SOLOHead` for default backend."""

    ctx = FUNCTION_REWRITER.get_context()
    batch_size = mlvl_cls_scores[0].size(0)
    cfg = self.test_cfg
    mlvl_cls_scores = [
        item.permute(0, 2, 3, 1).view(item.size(0), -1, self.cls_out_channels)
        for item in mlvl_cls_scores
    ]

    # avoid setting items
    lvl_strides = [
        torch.ones_like(mlvl_cls_scores[lvl][0, :, 0]) * self.strides[lvl]
        for lvl in range(len(mlvl_cls_scores))
    ]
    strides = torch.cat(lvl_strides, 0)
    assert len(mlvl_mask_preds) == len(mlvl_cls_scores)
    batch_mlvl_cls_scores = torch.cat(mlvl_cls_scores, dim=1)
    batch_mlvl_mask_preds = torch.cat(mlvl_mask_preds, dim=1)
    featmap_size = batch_mlvl_mask_preds.size()[-2:]
    batch_mlvl_cls_scores, cls_labels = torch.max(batch_mlvl_cls_scores, -1)

    score_mask = (batch_mlvl_cls_scores > cfg.score_thr)
    # pad zero to filter items
    batch_mlvl_cls_scores = batch_mlvl_cls_scores.where(
        score_mask, batch_mlvl_cls_scores.new_zeros(1)).view(-1)

    cls_labels = cls_labels.view(-1)

    mask_preds = batch_mlvl_mask_preds.view(-1, featmap_size[0],
                                            featmap_size[1])

    masks = (mask_preds > cfg.mask_thr)
    sum_masks = masks.sum((1, 2))
    keep = sum_masks > strides
    # pad zero to filter items
    cls_scores = batch_mlvl_cls_scores.where(
        keep, batch_mlvl_cls_scores.new_zeros(1))
    sum_masks = sum_masks.where(keep, sum_masks.new_ones(1))

    # maskness
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

    h, w = batch_img_metas[0]['img_shape'][:2]
    mask_preds = mask_preds[keep_inds].unsqueeze(0)

    mmdet_params = get_post_processing_params(ctx.cfg)
    export_postprocess_mask = mmdet_params.get('export_postprocess_mask', True)
    if export_postprocess_mask:
        upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)
        mask_preds = F.interpolate(
            mask_preds, size=upsampled_size, mode='bilinear')
        bboxes = scores.new_zeros(batch_size, scores.shape[-1], 4)
    else:

        bboxes = scores.new_zeros(batch_size, scores.shape[-1], 2)
        # full screen box so we can postprocess mask outside the model
        bboxes = torch.cat([
            bboxes,
            bboxes.new_full((*bboxes.shape[:2], 1), w),
            bboxes.new_full((*bboxes.shape[:2], 1), h)
        ],
                           dim=-1)

    labels = labels.reshape(batch_size, -1)
    dets = torch.cat([bboxes, scores.reshape(batch_size, -1, 1)], dim=-1)

    return dets, labels, mask_preds
