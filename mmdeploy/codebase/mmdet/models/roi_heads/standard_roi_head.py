# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.standard_roi_head.StandardRoIHead.simple_test')
def standard_roi_head__simple_test(ctx, self, x, proposals, img_metas,
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
            Each has shape (num_proposals, 5), last dimension
            5 represent (x1, y1, x2, y2, score).
        img_metas (list[dict]): Meta information of images.

    Returns:
        If self.with_mask == True:
            tuple[Tensor, Tensor, Tensor]: (det_bboxes, det_labels,
            segm_results), `det_bboxes` of shape [N, num_det, 5],
            `det_labels` of shape [N, num_det], and `segm_results`
            of shape [N, num_det, roi_H, roi_W].
        Else:
            tuple[Tensor, Tensor]: (det_bboxes, det_labels),
            `det_bboxes` of shape [N, num_det, 5] and `det_labels`
            of shape [N, num_det].
    """
    assert self.with_bbox, 'Bbox head must be implemented.'
    det_bboxes, det_labels = self.simple_test_bboxes(
        x, img_metas, proposals, self.test_cfg, rescale=False)
    if not self.with_mask:
        return det_bboxes, det_labels

    # padding zeros to det_bboxes and det_labels
    det_bboxes_tail = torch.zeros(
        det_bboxes.size(0),
        1,
        det_bboxes.size(2),
        device=det_bboxes.device,
        dtype=det_bboxes.dtype)
    det_labels_tail = torch.zeros(
        det_labels.size(0),
        1,
        device=det_labels.device,
        dtype=det_labels.dtype)
    det_bboxes = torch.cat([det_bboxes, det_bboxes_tail], 1)
    det_labels = torch.cat([det_labels, det_labels_tail], 1)

    segm_results = self.simple_test_mask(
        x, img_metas, det_bboxes, det_labels, rescale=False)
    return det_bboxes, det_labels, segm_results
