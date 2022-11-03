# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmdet.models.utils import (get_local_maximum, get_topk_from_heatmap,
                                transpose_and_gather_feat)
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.centernet_head.CenterNetHead.predict_by_feat')
def centernet_head__predict_by_feat__default(
        ctx,
        self,
        center_heatmap_preds: List[Tensor],
        wh_preds: List[Tensor],
        offset_preds: List[Tensor],
        batch_img_metas: List[dict],
        rescale: bool = True,
        with_nms: bool = False):
    """Rewrite `centernethead` of `CenterNetHead` for default backend."""

    # The dynamic shape deploy of CenterNet get wrong result on TensorRT-8.4.x
    # because of TensorRT bugs, https://github.com/NVIDIA/TensorRT/issues/2299,
    # FYI.

    assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
    batch_center_heatmap_preds = center_heatmap_preds[0]
    batch_wh_preds = wh_preds[0]
    batch_offset_preds = offset_preds[0]
    batch_size = batch_center_heatmap_preds.shape[0]
    img_shape = batch_img_metas[0]['img_shape']
    batch_det_bboxes, batch_labels = self._decode_heatmap(
        batch_center_heatmap_preds,
        batch_wh_preds,
        batch_offset_preds,
        img_shape,
        k=self.test_cfg.topk,
        kernel=self.test_cfg.local_maximum_kernel)
    det_bboxes = batch_det_bboxes.reshape([batch_size, -1, 5])
    det_labels = batch_labels.reshape(batch_size, -1)

    if with_nms:
        det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                  self.test_cfg)
    return det_bboxes, det_labels


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.centernet_head.CenterNetHead._decode_heatmap')
def centernet_head__decode_heatmap__default(
        ctx,
        self,
        center_heatmap_pred: Tensor,
        wh_pred: Tensor,
        offset_pred: Tensor,
        img_shape: tuple,
        k: int = 100,
        kernel: int = 3) -> Tuple[Tensor, Tensor]:
    """Rewrite `_decode_heatmap` of `CenterNetHead` for default backend."""

    # Rewrite this function to move the img_shape calculation outside the
    # model for dynamic shape deployment.
    height, width = center_heatmap_pred.shape[2:]
    inp_h, inp_w = img_shape
    center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)

    *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
        center_heatmap_pred, k=k)
    batch_scores, batch_index, batch_topk_labels = batch_dets

    wh = transpose_and_gather_feat(wh_pred, batch_index)
    offset = transpose_and_gather_feat(offset_pred, batch_index)
    topk_xs = topk_xs + offset[..., 0]
    topk_ys = topk_ys + offset[..., 1]
    tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
    tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
    br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
    br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

    batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
    batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
    return batch_bboxes, batch_topk_labels
