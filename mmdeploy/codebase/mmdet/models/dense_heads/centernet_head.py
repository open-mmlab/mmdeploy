# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.centernet_head.CenterNetHead.predict_by_feat')
def centernet_head__predict_by_feat__default(
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
