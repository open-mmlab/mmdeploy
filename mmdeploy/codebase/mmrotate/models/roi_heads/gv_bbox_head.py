# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch.nn.functional as F
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import InstanceList
from mmengine import ConfigDict
from mmrotate.structures.bbox import QuadriBoxes
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops import multiclass_nms


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.roi_heads.bbox_heads.GVBBoxHead.predict_by_feat')
def gv_bbox_head__predict_by_feat(self,
                                  rois: Tuple[Tensor],
                                  cls_scores: Tuple[Tensor],
                                  bbox_preds: Tuple[Tensor],
                                  fix_preds: Tuple[Tensor],
                                  ratio_preds: Tuple[Tensor],
                                  batch_img_metas: List[dict],
                                  rcnn_test_cfg: Optional[ConfigDict] = None,
                                  rescale: bool = False) -> InstanceList:
    """Transform network output for a batch into bbox predictions.

    Args:
        rois (tuple[Tensor]): Tuple of boxes to be transformed.
            Each has shape  (num_boxes, 5). last dimension 5 arrange as
            (batch_index, x1, y1, x2, y2).
        cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
            (num_boxes, num_classes + 1).
        bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
            has shape (num_boxes, num_classes * 4).
        fix_preds (tuple[Tensor]): Tuple of fix / deltas, each
            has shape (num_boxes, num_classes * 4).
        ratio_preds (tuple[Tensor]): Tuple of ratio / deltas, each
            has shape (num_boxes, num_classes * 1).
        batch_img_metas (list[dict]): List of image information.
        rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.

    Returns:
        list[:obj:`InstanceData`]: Instance segmentation
        results of each image after the post process.
        Each item usually contains following keys.

        - scores (Tensor): Classification scores, has a shape
          (num_instance, )
        - labels (Tensor): Labels of bboxes, has a shape
          (num_instances, ).
        - bboxes (Tensor): Has a shape (num_instances, 8),
          the last dimension 4 arrange as (x1, y1, ..., x4, y4).
    """
    assert rois.ndim == 3, 'Only support export two stage ' \
                           'model to ONNX ' \
                           'with batch dimension. '
    ctx = FUNCTION_REWRITER.get_context()

    img_shape = batch_img_metas[0]['img_shape']
    if self.custom_cls_channels:
        scores = self.loss_cls.get_activation(cls_scores)
    else:
        scores = F.softmax(
            cls_scores, dim=-1) if cls_scores is not None else None

    assert bbox_preds is not None
    bboxes = self.bbox_coder.decode(
        rois[..., 1:], bbox_preds, max_shape=img_shape)

    qboxes = self.fix_coder.decode(bboxes, fix_preds)

    bboxes = bboxes.view(*ratio_preds.size(), 4)
    qboxes = qboxes.view(*ratio_preds.size(), 8)

    from mmrotate.structures.bbox import hbox2qbox
    qboxes = qboxes.where(
        ratio_preds.unsqueeze(-1) < self.ratio_thr, hbox2qbox(bboxes))
    qboxes = qboxes.squeeze(2)

    bboxes = QuadriBoxes(qboxes)

    if self.predict_box_type == 'rbox':
        bboxes = bboxes.detach().convert_to('rbox')

    bboxes = get_box_tensor(bboxes)

    # ignore background class
    scores = scores[..., :self.num_classes]

    post_params = get_post_processing_params(ctx.cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = rcnn_test_cfg.nms.get('iou_threshold',
                                          post_params.iou_threshold)
    score_threshold = rcnn_test_cfg.get('score_thr',
                                        post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = rcnn_test_cfg.get('max_per_img', post_params.keep_top_k)

    nms_type = rcnn_test_cfg.nms.get('type')
    return multiclass_nms(
        bboxes,
        scores,
        max_output_boxes_per_class,
        nms_type=nms_type,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)
