# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmdet.utils import ConfigType
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.standard_roi_head.StandardRoIHead.predict_bbox')
def standard_roi_head__predict_bbox(self,
                                    x: Tuple[Tensor],
                                    batch_img_metas: List[dict],
                                    rpn_results_list: List[Tensor],
                                    rcnn_test_cfg: ConfigType,
                                    rescale: bool = False) -> List[Tensor]:
    """Rewrite `predict_bbox` of `StandardRoIHead` for default backend.

    Args:
        x (tuple[Tensor]): Feature maps of all scale level.
        batch_img_metas (list[dict]): List of image information.
        rpn_results_list (list[Tensor]): List of region
            proposals.
        rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.

    Returns:
        list[Tensor]: Detection results of each image
        after the post process.
        Each item usually contains following keys.

            - dets (Tensor): Classification bboxes and scores, has a shape
                (num_instance, 5)
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
    """
    rois = rpn_results_list[0]
    rois_dims = int(rois.shape[-1])
    batch_index = torch.arange(
        rois.shape[0], device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
    rois = torch.cat([batch_index, rois[..., :rois_dims - 1]], dim=-1)
    batch_size = rois.shape[0]
    num_proposals_per_img = rois.shape[1]

    # Eliminate the batch dimension
    rois = rois.view(-1, rois_dims)
    bbox_results = self._bbox_forward(x, rois)
    cls_scores = bbox_results['cls_score']
    bbox_preds = bbox_results['bbox_pred']

    # Recover the batch dimension
    rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
    cls_scores = cls_scores.reshape(batch_size, num_proposals_per_img,
                                    cls_scores.size(-1))

    bbox_preds = bbox_preds.reshape(batch_size, num_proposals_per_img,
                                    bbox_preds.size(-1))
    result_list = self.bbox_head.predict_by_feat(
        rois=rois,
        cls_scores=cls_scores,
        bbox_preds=bbox_preds,
        batch_img_metas=batch_img_metas,
        rcnn_test_cfg=rcnn_test_cfg,
        rescale=rescale)
    return result_list


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.standard_roi_head.StandardRoIHead.predict_mask')
def standard_roi_head__predict_mask(self,
                                    x: Tuple[Tensor],
                                    batch_img_metas: List[dict],
                                    results_list: List[Tensor],
                                    rescale: bool = False) -> List[Tensor]:
    """Perform forward propagation of the mask head and predict detection
    results on the features of the upstream network.

    Args:
        x (tuple[Tensor]): Feature maps of all scale level.
        batch_img_metas (list[dict]): List of image information.
        results_list (list[:obj:`InstanceData`]): Detection results of
            each image.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.

    Returns:
        list[Tensor]: Detection results of each image
        after the post process.
        Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
                the last dimension 4 arrange as (x1, y1, x2, y2).
            - masks (Tensor): Has a shape (num_instances, H, W).
    """
    dets, det_labels = results_list
    batch_size = dets.size(0)
    det_bboxes = dets[..., :4]
    # expand might lead to static shape, use broadcast instead
    batch_index = torch.arange(
        det_bboxes.size(0), device=det_bboxes.device).float().view(
            -1, 1, 1) + det_bboxes.new_zeros(
                (det_bboxes.size(0), det_bboxes.size(1))).unsqueeze(-1)
    mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
    mask_rois = mask_rois.view(-1, 5)
    mask_results = self._mask_forward(x, mask_rois)
    mask_preds = mask_results['mask_preds']
    num_det = det_bboxes.shape[1]
    segm_results = self.mask_head.predict_by_feat(
        mask_preds,
        results_list,
        batch_img_metas,
        self.test_cfg,
        rescale=rescale)
    segm_results = segm_results.reshape(batch_size, num_det,
                                        segm_results.shape[-2],
                                        segm_results.shape[-1])
    return dets, det_labels, segm_results
