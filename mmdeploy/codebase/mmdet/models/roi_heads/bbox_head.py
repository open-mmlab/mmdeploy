# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmdet.structures.bbox import get_box_tensor
from mmengine import ConfigDict
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops import multiclass_nms


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.bbox_heads.bbox_head.BBoxHead.forward')
@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.ConvFCBBoxHead.forward'
)
def bbox_head__forward(self, x):
    """Rewrite `forward` for default backend.

    This function uses the specific `forward` function for the BBoxHead
    or ConvFCBBoxHead after adding marks.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        x (Tensor): Input image tensor.

    Returns:
        tuple(Tensor, Tensor): The (cls_score, bbox_pred). The cls_score
        has shape (N, num_det, num_cls) and the bbox_pred has shape
        (N, num_det, 4).
    """
    ctx = FUNCTION_REWRITER.get_context()

    @mark(
        'bbox_head_forward',
        inputs=['bbox_feats'],
        outputs=['cls_score', 'bbox_pred'])
    def __forward(self, x):
        return ctx.origin_func(self, x)

    return __forward(self, x)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.bbox_heads.bbox_head.BBoxHead.predict_by_feat')
def bbox_head__predict_by_feat(self,
                               rois: Tuple[Tensor],
                               cls_scores: Tuple[Tensor],
                               bbox_preds: Tuple[Tensor],
                               batch_img_metas: List[dict],
                               rcnn_test_cfg: Optional[ConfigDict] = None,
                               rescale: bool = False) -> Tuple[Tensor]:
    """Rewrite `predict_by_feat` of `BBoxHead` for default backend.

    Transform network output for a batch into bbox predictions. Support
    `reg_class_agnostic == False` case.

    Args:
        rois (tuple[Tensor]): Tuple of boxes to be transformed.
            Each has shape  (num_boxes, 5). last dimension 5 arrange as
            (batch_index, x1, y1, x2, y2).
        cls_scores (tuple[Tensor]): Tuple of box scores, each has shape
            (num_boxes, num_classes + 1).
        bbox_preds (tuple[Tensor]): Tuple of box energies / deltas, each
            has shape (num_boxes, num_classes * 4).
        batch_img_metas (list[dict]): List of image information.
        rcnn_test_cfg (obj:`ConfigDict`, optional): `test_cfg` of R-CNN.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.

    Returns:
            - dets (Tensor): Classification bboxes and scores, has a shape
                (num_instance, 5)
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
    """
    ctx = FUNCTION_REWRITER.get_context()
    assert rois.ndim == 3, 'Only support export two stage ' \
                           'model to ONNX ' \
                           'with batch dimension. '

    img_shape = batch_img_metas[0]['img_shape']
    if self.custom_cls_channels:
        scores = self.loss_cls.get_activation(cls_scores)
    else:
        scores = F.softmax(
            cls_scores, dim=-1) if cls_scores is not None else None

    if bbox_preds is not None:
        # num_classes = 1 if self.reg_class_agnostic else self.num_classes
        # if num_classes > 1:
        #     rois = rois.repeat_interleave(num_classes, dim=1)
        bboxes = self.bbox_coder.decode(
            rois[..., 1:], bbox_preds, max_shape=img_shape)
        bboxes = get_box_tensor(bboxes)
    else:
        bboxes = rois[..., 1:].clone()
        if img_shape is not None:
            max_shape = bboxes.new_tensor(img_shape)[..., :2]
            min_xy = bboxes.new_tensor(0)
            max_xy = torch.cat([max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
            bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
            bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    batch_size = scores.shape[0]
    device = scores.device
    # ignore background class
    scores = scores[..., :self.num_classes]
    if not self.reg_class_agnostic:
        # only keep boxes with the max scores
        max_inds = scores.reshape(-1, self.num_classes).argmax(1, keepdim=True)
        encode_size = self.bbox_coder.encode_size
        bboxes = bboxes.reshape(-1, self.num_classes, encode_size)
        dim0_inds = torch.arange(bboxes.shape[0], device=device).unsqueeze(-1)
        bboxes = bboxes[dim0_inds, max_inds].reshape(batch_size, -1,
                                                     encode_size)
    # get nms params
    post_params = get_post_processing_params(ctx.cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = rcnn_test_cfg.nms.get('iou_threshold',
                                          post_params.iou_threshold)
    score_threshold = rcnn_test_cfg.get('score_thr',
                                        post_params.score_threshold)
    if torch.onnx.is_in_onnx_export():
        pre_top_k = post_params.pre_top_k
    else:
        # For two stage partition post processing
        pre_top_k = -1 if post_params.pre_top_k >= bboxes.shape[1] \
            else post_params.pre_top_k
    keep_top_k = rcnn_test_cfg.get('max_per_img', post_params.keep_top_k)
    nms_type = rcnn_test_cfg.nms.get('type')
    dets, labels = multiclass_nms(
        bboxes,
        scores,
        max_output_boxes_per_class,
        nms_type=nms_type,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)

    return dets, labels
