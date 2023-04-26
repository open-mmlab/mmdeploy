# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn.functional as F
from mmdet.models.layers.matrix_nms import mask_matrix_nms
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.solov2_head.'
    'SOLOV2Head.predict_by_feat')
def solov2_head__predict_by_feat(self, mlvl_kernel_preds: List[Tensor],
                                 mlvl_cls_scores: List[Tensor],
                                 mask_feats: Tensor,
                                 batch_img_metas: List[Dict], **kwargs):
    """Rewrite `predict_by_feat` of `SOLOV2Head` for default backend.

    Args:
        mlvl_kernel_preds (list[Tensor]): Multi-level dynamic kernel
            prediction. The kernel is used to generate instance
            segmentation masks by dynamic convolution. Each element in the
            list has shape
            (batch_size, kernel_out_channels, num_grids, num_grids).
        mlvl_cls_scores (list[Tensor]): Multi-level scores. Each element
            in the list has shape
            (batch_size, num_classes, num_grids, num_grids).
        mask_feats (Tensor): Unified mask feature map used to generate
            instance segmentation masks by dynamic convolution. Has shape
            (batch_size, mask_out_channels, h, w).
        batch_img_metas (list[dict]): Meta information of all images.

    Returns:
        list[:obj:`InstanceData`]: Processed results of multiple
        images.Each :obj:`InstanceData` usually contains
        following keys.

            - scores (Tensor): Classification scores, has shape
                (num_instance,).
            - labels (Tensor): Has shape (num_instances,).
            - masks (Tensor): Processed mask results, has
                shape (num_instances, h, w).
    """
    ctx = FUNCTION_REWRITER.get_context()
    cfg = self.test_cfg
    num_levels = len(mlvl_cls_scores)
    batch_size = mlvl_cls_scores[0].size(0)
    assert len(mlvl_kernel_preds) == len(mlvl_cls_scores)

    for lvl in range(num_levels):
        kernel_preds = mlvl_kernel_preds[lvl]
        cls_scores = mlvl_cls_scores[lvl]
        cls_scores = cls_scores.sigmoid()
        local_max = F.max_pool2d(cls_scores, 2, stride=1, padding=1)
        keep_mask = local_max[:, :, :-1, :-1] == cls_scores
        cls_scores = cls_scores * keep_mask
        mlvl_cls_scores[lvl] = cls_scores.permute(0, 2, 3, 1).view(
            batch_size, -1, self.cls_out_channels)
        mlvl_kernel_preds[lvl] = kernel_preds.permute(0, 2, 3, 1).view(
            batch_size, -1, self.kernel_out_channels)

    # Rewrite strides to avoid set_items.
    mlvl_strides = [
        torch.ones_like(mlvl_cls_scores[lvl][0, :, 0]) * self.strides[lvl]
        for lvl in range(len(mlvl_cls_scores))
    ]
    strides = torch.cat(mlvl_strides, 0)
    assert len(mlvl_kernel_preds) == len(mlvl_cls_scores)
    batch_mlvl_cls_scores = torch.cat(mlvl_cls_scores, dim=1)
    batch_mlvl_kernel_preds = torch.cat(mlvl_kernel_preds, dim=1)

    featmap_size = mask_feats.size()[-2:]
    h, w = batch_img_metas[0]['img_shape'][:2]
    batch_mlvl_cls_scores, cls_labels = torch.max(batch_mlvl_cls_scores, -1)

    score_mask = (batch_mlvl_cls_scores > cfg.score_thr)
    batch_mlvl_cls_scores = batch_mlvl_cls_scores.where(
        score_mask, batch_mlvl_cls_scores.new_zeros(1)).view(-1)

    cls_labels = cls_labels.view(-1)

    # mask encoding.

    kernel_preds = batch_mlvl_kernel_preds[0].unsqueeze(2).unsqueeze(3)
    mask_preds = F.conv2d(
        mask_feats, kernel_preds, stride=1).squeeze(0).sigmoid()
    aligned_score_mask = score_mask[0].unsqueeze(1).unsqueeze(2)
    mask_preds = mask_preds.where(aligned_score_mask, mask_preds.new_zeros(1))

    # mask.
    masks = (mask_preds > cfg.mask_thr)
    sum_masks = masks.sum((1, 2))
    keep = sum_masks > strides
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

    mask_preds = mask_preds[keep_inds].unsqueeze(0)
    post_params = get_post_processing_params(ctx.cfg)
    export_postprocess_mask = post_params.get('export_postprocess_mask', True)
    if export_postprocess_mask:
        upsampled_size = (featmap_size[0] * self.mask_stride,
                          featmap_size[1] * self.mask_stride)
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
