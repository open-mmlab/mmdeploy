import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.test_mixins.\
     BBoxTestMixin.simple_test_bboxes')
def simple_test_bboxes_of_bbox_test_mixin(ctx, self, x, img_metas, proposals,
                                          rcnn_test_cfg, **kwargs):
    """Rewrite `simple_test_bboxes` for default backend."""
    rois = proposals
    batch_index = torch.arange(
        rois.shape[0], device=rois.device).float().view(-1, 1, 1).expand(
            rois.size(0), rois.size(1), 1)
    rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
    batch_size = rois.shape[0]
    num_proposals_per_img = rois.shape[1]

    # Eliminate the batch dimension
    rois = rois.view(-1, 5)
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
        rois, cls_score, bbox_pred, img_metas['img_shape'], cfg=rcnn_test_cfg)
    return det_bboxes, det_labels


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.test_mixins.\
    MaskTestMixin.simple_test_mask')
def simple_test_mask_of_mask_test_mixin(ctx, self, x, img_metas, det_bboxes,
                                        det_labels, **kwargs):
    """Rewrite `simple_test_mask` for default backend."""
    if det_bboxes.shape[1] == 0:
        bboxes_shape, labels_shape = list(det_bboxes.shape), list(
            det_labels.shape)
        bboxes_shape[1], labels_shape[1] = 1, 1
        det_bboxes = torch.tensor(
            [[[0., 0., 1., 1., 0.]]],
            device=det_bboxes.device).expand(*bboxes_shape)
        det_labels = torch.tensor(
            [[0]], device=det_bboxes.device).expand(*labels_shape)

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
    max_shape = img_metas['img_shape']
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
