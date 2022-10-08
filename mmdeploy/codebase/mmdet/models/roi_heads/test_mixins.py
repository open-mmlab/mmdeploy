# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.test_mixins.BBoxTestMixin.simple_test_bboxes')
def bbox_test_mixin__simple_test_bboxes(ctx,
                                        self,
                                        x,
                                        img_metas,
                                        proposals,
                                        rcnn_test_cfg,
                                        rescale=False):
    """Rewrite `simple_test_bboxes` of `BBoxTestMixin` for default backend.

    1. This function eliminates the batch dimension to get forward bbox
    results, and recover batch dimension to calculate final result
    for deployment.
    2. This function returns detection result as Tensor instead of numpy
    array.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        x (tuple[Tensor]): Features from upstream network. Each
            has shape (batch_size, c, h, w).
        img_metas (list[dict]): Meta information of images.
        proposals (list(Tensor)): Proposals from rpn head.
            Each has shape (num_proposals, 5), last dimension
            5 represent (x1, y1, x2, y2, score).
        rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
        rescale (bool): If True, return boxes in original image space.
            Default: False.

    Returns:
        tuple[Tensor, Tensor]: (det_bboxes, det_labels), `det_bboxes` of
        shape [N, num_det, 5] and `det_labels` of shape [N, num_det].
    """
    rois = proposals
    batch_index = torch.arange(
        rois.shape[0], device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
    rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
    batch_size = rois.shape[0]

    # Eliminate the batch dimension
    rois = rois.view(-1, 5)
    bbox_results = self._bbox_forward(x, rois)
    cls_score = bbox_results['cls_score']
    bbox_pred = bbox_results['bbox_pred']

    # Recover the batch dimension
    rois = rois.reshape(batch_size, -1, rois.size(-1))
    cls_score = cls_score.reshape(batch_size, -1, cls_score.size(-1))

    bbox_pred = bbox_pred.reshape(batch_size, -1, bbox_pred.size(-1))
    det_bboxes, det_labels = self.bbox_head.get_bboxes(
        rois,
        cls_score,
        bbox_pred,
        img_metas[0]['img_shape'],
        None,
        rescale=rescale,
        cfg=rcnn_test_cfg)
    return det_bboxes, det_labels


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.test_mixins.MaskTestMixin.simple_test_mask')
def mask_test_mixin__simple_test_mask(ctx, self, x, img_metas, det_bboxes,
                                      det_labels, **kwargs):
    """Rewrite `simple_test_mask` of `BBoxTestMixin` for default backend.

    This function returns detection result as Tensor instead of numpy
    array.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        x (tuple[Tensor]): Features from upstream network. Each
            has shape (batch_size, c, h, w).
        img_metas (list[dict]): Meta information of images.
        det_bboxes (tuple[Tensor]): Detection bounding-boxes from features.
        Each has shape of (batch_size, num_det, 5).
        det_labels (tuple[Tensor]): Detection labels from features. Each
        has shape of (batch_size, num_det).

    Returns:
        tuple[Tensor]: (segm_results), `segm_results` of shape
        [N, num_det, roi_H, roi_W].
    """
    batch_size = det_bboxes.size(0)
    det_bboxes = det_bboxes[..., :4]
    batch_index = torch.arange(
        det_bboxes.size(0),
        device=det_bboxes.device).float().view(-1, 1, 1).expand(
            det_bboxes.size(0), det_bboxes.size(1), 1)
    mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
    mask_rois = mask_rois.view(-1, 5)
    mask_results = self._mask_forward(x, mask_rois)
    mask_pred = mask_results['mask_pred']
    max_shape = img_metas[0]['img_shape']
    num_det = det_bboxes.shape[1]
    det_bboxes = det_bboxes.reshape(-1, 4)
    det_labels = det_labels.reshape(-1)
    segm_results = self.mask_head.get_seg_masks(mask_pred, det_bboxes,
                                                det_labels, self.test_cfg,
                                                max_shape)
    segm_results = segm_results.reshape(batch_size, num_det,
                                        segm_results.shape[-2],
                                        segm_results.shape[-1])
    return segm_results
