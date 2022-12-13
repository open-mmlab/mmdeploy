# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmdet.utils import ConfigType, InstanceList
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.roi_heads.gv_ratio_roi_head'
    '.GVRatioRoIHead.predict_bbox')
def gv_ratio_roi_head__predict_bbox(self,
                                    x: Tuple[Tensor],
                                    batch_img_metas: List[dict],
                                    rpn_results_list: InstanceList,
                                    rcnn_test_cfg: ConfigType,
                                    rescale: bool = False):
    """Test only det bboxes without augmentation.

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
    batch_index = torch.arange(
        rois.shape[0], device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
    rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
    batch_size = rois.shape[0]
    num_proposals_per_img = rois.shape[1]

    # Eliminate the batch dimension
    rois = rois.view(-1, 5)
    bbox_results = self._bbox_forward(x, rois)
    cls_scores = bbox_results['cls_score']
    bbox_preds = bbox_results['bbox_pred']
    fix_preds = bbox_results['fix_pred']
    ratio_preds = bbox_results['ratio_pred']

    # Recover the batch dimension
    rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
    cls_scores = cls_scores.reshape(batch_size, num_proposals_per_img,
                                    cls_scores.size(-1))

    bbox_preds = bbox_preds.reshape(batch_size, num_proposals_per_img,
                                    bbox_preds.size(-1))
    fix_preds = fix_preds.reshape(batch_size, num_proposals_per_img,
                                  fix_preds.size(-1))
    ratio_preds = ratio_preds.reshape(batch_size, num_proposals_per_img,
                                      ratio_preds.size(-1))
    result_list = self.bbox_head.predict_by_feat(
        rois=rois,
        cls_scores=cls_scores,
        bbox_preds=bbox_preds,
        fix_preds=fix_preds,
        ratio_preds=ratio_preds,
        batch_img_metas=batch_img_metas,
        rcnn_test_cfg=rcnn_test_cfg,
        rescale=rescale)
    return result_list
