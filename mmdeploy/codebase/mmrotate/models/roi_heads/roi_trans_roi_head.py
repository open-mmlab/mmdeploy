# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.roi_heads.roi_trans_roi_head'
    '.RoITransRoIHead.simple_test')
def roi_trans_roi_head__simple_test(ctx, self, x, proposal_list, img_metas,
                                    **kwargs):
    """Rewrite `simple_test` of `RoITransRoIHead` for default backend.

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

    rois, labels = proposal_list
    assert rois.shape[0] == 1, ('Only support one input image '
                                'while in exporting to ONNX')
    # Remove the scores
    rois = rois[..., :-1]
    batch_size = rois.shape[0]
    num_proposals_per_img = rois.shape[1]
    # Eliminate the batch dimension
    # Note that first RoIs in RoITransformer are horizontal bounding boxes.
    rois = rois.view(-1, 4)

    # Add dummy batch index
    rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)

    max_shape = img_metas[0]['img_shape']
    ms_scores = []
    rcnn_test_cfg = self.test_cfg

    for i in range(self.num_stages):
        bbox_results = self._bbox_forward(i, x, rois)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))

        ms_scores.append(cls_score)

        if i < self.num_stages - 1:
            assert self.bbox_head[i].reg_class_agnostic
            new_rois = self.bbox_head[i].bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=max_shape)
            rois = new_rois.reshape(-1, new_rois.shape[-1])
            # Add dummy batch index
            rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)

    # average scores of each image by stages
    cls_score = sum(ms_scores) / float(len(ms_scores))
    bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                  bbox_pred.size(-1))
    rois = rois.reshape(batch_size, num_proposals_per_img, -1)

    scale_factor = img_metas[0].get('scale_factor', None)
    det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
        rois, cls_score, bbox_pred, max_shape, scale_factor, cfg=rcnn_test_cfg)

    return det_bboxes, det_labels
