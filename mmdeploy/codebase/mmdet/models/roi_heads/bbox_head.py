# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdeploy.codebase.mmdet.core.post_processing import multiclass_nms
from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER, mark


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.bbox_heads.bbox_head.BBoxHead.forward')
@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.ConvFCBBoxHead.forward'
)
@mark(
    'bbox_head_forward',
    inputs=['bbox_feats'],
    outputs=['cls_score', 'bbox_pred'])
def bbox_head__forward(ctx, self, x):
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
    return ctx.origin_func(self, x)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.bbox_heads.bbox_head.BBoxHead.get_bboxes')
def bbox_head__get_bboxes(ctx,
                          self,
                          rois,
                          cls_score,
                          bbox_pred,
                          img_shape,
                          scale_factor,
                          rescale=False,
                          cfg=None):
    """Rewrite `get_bboxes` of `bbox_head` for default backend.

    Transform network output for a batch into bbox predictions. Support
    `reg_class_agnostic == False` case.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (ATSSHead): The instance of the class ATSSHead.
        rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
            last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
        cls_score (Tensor): Box scores, has shape
            (num_boxes, num_classes + 1).
        bbox_pred (Tensor, optional): Box energies / deltas.
            has shape (num_boxes, num_classes * 4).
        img_shape (Sequence[int], optional): Maximum bounds for boxes,
            specifies (H, W, C) or (H, W).
        cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None


    Returns:
        tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
        `dets` of shape [N, num_det, 5] and `labels` of shape
        [N, num_det].
    """
    assert rois.ndim == 3, 'Only support export two stage ' \
                           'model to ONNX ' \
                           'with batch dimension. '
    if self.custom_cls_channels:
        scores = self.loss_cls.get_activation(cls_score)
    else:
        scores = F.softmax(
            cls_score, dim=-1) if cls_score is not None else None

    if bbox_pred is not None:
        bboxes = self.bbox_coder.decode(
            rois[..., 1:], bbox_pred, max_shape=img_shape)
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
        bboxes = bboxes.reshape(-1, self.num_classes, 4)
        dim0_inds = torch.arange(bboxes.shape[0], device=device).unsqueeze(-1)
        bboxes = bboxes[dim0_inds, max_inds].reshape(batch_size, -1, 4)

    # get nms params
    post_params = get_post_processing_params(ctx.cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    if torch.onnx.is_in_onnx_export():
        pre_top_k = post_params.pre_top_k
    else:
        # For two stage partition post processing
        pre_top_k = -1 if post_params.pre_top_k >= bboxes.shape[1] \
            else post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    dets, labels = multiclass_nms(
        bboxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)

    return dets, labels
