from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.roi_heads.StandardRoIHead.simple_test')
def simple_test_of_standard_roi_head(ctx, self, x, proposals, img_metas,
                                     **kwargs):
    """Rewrite `simple_test` for default backend."""
    assert self.with_bbox, 'Bbox head must be implemented.'
    det_bboxes, det_labels = self.simple_test_bboxes(
        x, img_metas, proposals, self.test_cfg, rescale=False)
    if not self.with_mask:
        return det_bboxes, det_labels

    segm_results = self.simple_test_mask(
        x, img_metas, det_bboxes, det_labels, rescale=False)
    return det_bboxes, det_labels, segm_results
