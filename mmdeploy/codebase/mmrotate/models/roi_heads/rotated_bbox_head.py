# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.codebase.mmrotate.core.post_processing import \
    multiclass_nms_rotated
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.roi_heads.bbox_heads.RotatedBBoxHead.get_bboxes')
def rotated_bbox_head__get_bboxes(ctx,
                                  self,
                                  rois,
                                  cls_score,
                                  bbox_pred,
                                  img_shape,
                                  scale_factor,
                                  rescale=False,
                                  cfg=None):
    """Transform network output for a batch into bbox predictions.

    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape
            (num_boxes, 6). last dimension 5 arrange as
            (batch_index, x, y, w, h, theta).
        cls_score (torch.Tensor): Box scores, has shape
            (num_boxes, num_classes + 1).
        bbox_pred (Tensor, optional): Box energies / deltas.
            has shape (num_boxes, num_classes * 6).
        img_shape (Sequence[int], optional): Maximum bounds for boxes,
            specifies (H, W, C) or (H, W).
        scale_factor (ndarray): Scale factor of the
           image arrange as (w_scale, h_scale, w_scale, h_scale).
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

    Returns:
        tuple[Tensor, Tensor]:
            First tensor is `det_bboxes`, has the shape
            (num_boxes, 6) and last
            dimension 6 represent (cx, cy, w, h, theta, score).
            Second tensor is the labels with shape (num_boxes, ).
    """
    assert rois.ndim == 3, 'Only support export two stage ' \
                           'model to ONNX ' \
                           'with batch dimension. '

    if self.custom_cls_channels:
        scores = self.loss_cls.get_activation(cls_score)
    else:
        scores = F.softmax(
            cls_score, dim=-1) if cls_score is not None else None

    assert bbox_pred is not None
    bboxes = self.bbox_coder.decode(
        rois[..., 1:], bbox_pred, max_shape=img_shape)

    batch_size = scores.shape[0]
    device = scores.device
    # ignore background class
    scores = scores[..., :self.num_classes]
    if not self.reg_class_agnostic:
        # only keep boxes with the max scores
        max_inds = scores.reshape(-1, self.num_classes).argmax(1, keepdim=True)
        bboxes = bboxes.reshape(-1, self.num_classes, 5)
        dim0_inds = torch.arange(bboxes.shape[0], device=device).unsqueeze(-1)
        bboxes = bboxes[dim0_inds, max_inds].reshape(batch_size, -1, 5)

    post_params = get_post_processing_params(ctx.cfg)
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    return multiclass_nms_rotated(
        bboxes,
        scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)
