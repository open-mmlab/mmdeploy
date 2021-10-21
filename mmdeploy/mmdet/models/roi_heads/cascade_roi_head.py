import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.CascadeRoIHead.simple_test')
def simple_test_of_cascade_roi_head(ctx, self, x, proposals, img_metas,
                                    **kwargs):
    """Rewrite `simple_test` for default backend."""
    assert self.with_bbox, 'Bbox head must be implemented.'
    assert proposals.shape[0] == 1, 'Only support one input image ' \
                                    'while in exporting to ONNX'
    # Remove the scores
    rois = proposals[..., :-1]
    batch_size = rois.shape[0]
    num_proposals_per_img = rois.shape[1]
    # Eliminate the batch dimension
    rois = rois.view(-1, 4)

    # Add dummy batch index
    rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)

    max_shape = img_metas['img_shape']
    ms_scores = []
    rcnn_test_cfg = self.test_cfg

    for i in range(self.num_stages):
        bbox_results = self._bbox_forward(i, x, rois)

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
        ms_scores.append(cls_score)
        if i < self.num_stages - 1:
            assert self.bbox_head[i].reg_class_agnostic
            new_rois = self.bbox_head[i].bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=max_shape)
            rois = new_rois.reshape(-1, new_rois.shape[-1])
            # Add dummy batch index
            rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)

    cls_score = sum(ms_scores) / float(len(ms_scores))
    bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
    rois = rois.reshape(batch_size, num_proposals_per_img, -1)
    det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
        rois, cls_score, bbox_pred, max_shape, cfg=rcnn_test_cfg)

    if not self.with_mask:
        return det_bboxes, det_labels
    else:
        raise NotImplementedError(
            'Masks are not yet supported in export CascadeRoIHead.simple_test')
