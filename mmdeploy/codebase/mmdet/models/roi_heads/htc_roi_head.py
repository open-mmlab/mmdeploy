# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.htc_roi_head.HybridTaskCascadeRoIHead.predict_mask'
)
def htc_roi_head__predict_mask(self,
                               x: Tuple[Tensor],
                               semantic_heat: Tensor,
                               batch_img_metas: List[dict],
                               results_list: List[Tensor],
                               rescale: bool = False) -> List[Tensor]:
    dets, det_labels = results_list

    batch_size = dets.size(0)
    det_bboxes = dets[..., :4]
    batch_index = torch.arange(
        det_bboxes.size(0),
        device=det_bboxes.device).float().view(-1, 1, 1).expand(
            det_bboxes.size(0), det_bboxes.size(1), 1)
    mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
    mask_rois = mask_rois.view(-1, 5)

    mask_results = self._mask_forward(
        stage=-1,
        x=x,
        rois=mask_rois,
        semantic_feat=semantic_heat,
        training=False)

    mask_preds = mask_results['mask_preds'][0]
    num_det = det_bboxes.shape[1]
    segm_results = self.mask_head[-1].predict_by_feat(
        mask_preds,
        results_list,
        batch_img_metas,
        self.test_cfg,
        rescale=rescale)
    segm_results = segm_results.reshape(batch_size, num_det,
                                        segm_results.shape[-2],
                                        segm_results.shape[-1])
    return dets, det_labels, segm_results
