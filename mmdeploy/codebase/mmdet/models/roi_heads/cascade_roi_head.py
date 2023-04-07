# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.roi_heads.cascade_roi_head.CascadeRoIHead.simple_test')
def cascade_roi_head__simple_test(ctx, self, x, proposals, img_metas,
                                  **kwargs):
    """Rewrite `simple_test` of `CascadeRoIHead` for default backend.

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
    assert proposals.shape[0] == 1, 'Only support one input image ' \
                                    'while in exporting to ONNX'
    # Remove the scores
    rois = proposals[..., :-1]
    num_proposals_per_img = rois.shape[1]
    batch_size = rois.shape[0]
    # Eliminate the batch dimension
    rois = rois.view(-1, 4)
    inds = torch.arange(
        batch_size, device=rois.device).float().repeat(num_proposals_per_img,
                                                       1)
    inds = inds.t().reshape(-1, 1)
    rois = torch.cat([inds, rois], dim=1)

    max_shape = None
    scale_factor = None
    ms_scores = []
    rcnn_test_cfg = self.test_cfg

    for i in range(self.num_stages):
        bbox_results = self._bbox_forward(i, x, rois)

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        ms_scores.append(cls_score)
        if i < self.num_stages - 1:
            assert not self.bbox_head[i].custom_activation
            bbox_label = cls_score[:, :-1].argmax(dim=1)
            rois = self.bbox_head[i].regress_by_class(rois, bbox_label,
                                                      bbox_pred, img_metas[0])

    cls_score = sum(ms_scores) / float(len(ms_scores))
    cls_score = cls_score.reshape(batch_size, -1, cls_score.size(-1))
    rois = rois.reshape(batch_size, -1, rois.size(-1))
    bbox_pred = bbox_pred.reshape(batch_size, -1, bbox_pred.size(-1))

    det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
        rois, cls_score, bbox_pred, max_shape, scale_factor, cfg=rcnn_test_cfg)

    if not self.with_mask:
        return det_bboxes, det_labels
    else:
        batch_index = torch.arange(det_bboxes.size(0),
                                   device=det_bboxes.device). \
            float().view(-1, 1, 1).expand(
            det_bboxes.size(0), det_bboxes.size(1), 1)
        rois = det_bboxes[..., :4]
        mask_rois = torch.cat([batch_index, rois], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        aug_masks = []
        for i in range(self.num_stages):
            mask_results = self._mask_forward(i, x, mask_rois)
            mask_pred = mask_results['mask_pred']
            aug_masks.append(mask_pred)
        # Calculate the mean of masks from several stage
        mask_pred = sum(aug_masks) / len(aug_masks)
        segm_results = self.mask_head[-1].get_seg_masks(
            mask_pred, rois.reshape(-1, 4), det_labels.reshape(-1),
            self.test_cfg, max_shape)
        segm_results = segm_results.reshape(batch_size, det_bboxes.shape[1],
                                            segm_results.shape[-2],
                                            segm_results.shape[-1])
        return det_bboxes, det_labels, segm_results
