# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.roi_heads.rotate_standard_roi_head'
    '.RotatedStandardRoIHead.simple_test')
def rotate_standard_roi_head__simple_test(ctx, self, x, proposals, img_metas,
                                          **kwargs):
    """Rewrite `simple_test` of `StandardRoIHead` for default backend.

    This function returns detection result as Tensor instead of numpy
    array.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        x (tuple[Tensor]): Features from upstream network. Each
            has shape (batch_size, c, h, w).
        proposals (list(Tensor)): Proposals from rpn head.
            Each has shape (num_proposals, 6), last dimension
            6 represent (x, y, w, h, theta, score).
        img_metas (list[dict]): Meta information of images.

    Returns:
        tuple[Tensor, Tensor]: (det_bboxes, det_labels),
        `det_bboxes` of shape [N, num_det, 6] and `det_labels`
        of shape [N, num_det].
    """
    assert self.with_bbox, 'Bbox head must be implemented.'
    det_bboxes, det_labels = self.simple_test_bboxes(
        x, img_metas, proposals, self.test_cfg, rescale=False)

    return det_bboxes, det_labels


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.roi_heads.oriented_standard_roi_head'
    '.OrientedStandardRoIHead.simple_test_bboxes')
def oriented_standard_roi_head__simple_test_bboxes(ctx,
                                                   self,
                                                   x,
                                                   img_metas,
                                                   proposals,
                                                   rcnn_test_cfg,
                                                   rescale=False):
    """Test only det bboxes without augmentation.

    Args:
        x (tuple[Tensor]): Feature maps of all scale level.
        img_metas (list[dict]): Image meta info.
        proposals (List[Tensor]): Region proposals.
        rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
        rescale (bool): If True, return boxes in original image space.
            Default: False.

    Returns:
        tuple[list[Tensor], list[Tensor]]: The first list contains \
            the boxes of the corresponding image in a batch, each \
            tensor has the shape (num_boxes, 6) and last dimension \
            6 represent (x, y, w, h, theta, score). Each Tensor \
            in the second list is the labels with shape (num_boxes, ). \
            The length of both lists should be equal to batch_size.
    """

    rois, labels = proposals
    batch_index = torch.arange(
        rois.shape[0], device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
    rois = torch.cat([batch_index, rois[..., :5]], dim=-1)
    batch_size = rois.shape[0]
    num_proposals_per_img = rois.shape[1]

    # Eliminate the batch dimension
    rois = rois.view(-1, 6)
    bbox_results = self._bbox_forward(x, rois)
    cls_score = bbox_results['cls_score']
    bbox_pred = bbox_results['bbox_pred']

    # Recover the batch dimension
    rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
    cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                  cls_score.size(-1))

    bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                  bbox_pred.size(-1))
    det_bboxes, det_labels = self.bbox_head.get_bboxes(
        rois,
        cls_score,
        bbox_pred,
        img_metas[0]['img_shape'],
        None,
        rescale=rescale,
        cfg=rcnn_test_cfg)
    return det_bboxes, det_labels
