# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.codebase.mmrotate.core.post_processing import \
    multiclass_nms_rotated
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.roi_heads.bbox_heads.GVBBoxHead.get_bboxes')
def gv_bbox_head__get_bboxes(ctx,
                             self,
                             rois,
                             cls_score,
                             bbox_pred,
                             fix_pred,
                             ratio_pred,
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

    rbboxes = self.fix_coder.decode(bboxes, fix_pred)

    bboxes = bboxes.view(*ratio_pred.size(), 4)
    rbboxes = rbboxes.view(*ratio_pred.size(), 5)

    from mmrotate.core import hbb2obb
    rbboxes = rbboxes.where(
        ratio_pred.unsqueeze(-1) < self.ratio_thr,
        hbb2obb(bboxes, self.version))
    rbboxes = rbboxes.squeeze(2)

    # ignore background class
    scores = scores[..., :self.num_classes]

    post_params = get_post_processing_params(ctx.cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    return multiclass_nms_rotated(
        rbboxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)
