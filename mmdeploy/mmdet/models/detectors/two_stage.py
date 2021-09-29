from mmdeploy.core import FUNCTION_REWRITER, mark


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.TwoStageDetector.extract_feat')
@mark('extract_feat', inputs='img', outputs='feat')
def extract_feat_of_two_stage(ctx, self, img):
    """Rewrite `extract_feat` for default backend."""
    return ctx.origin_func(self, img)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.TwoStageDetector.simple_test')
def simple_test_of_two_stage(ctx,
                             self,
                             img,
                             img_metas,
                             proposals=None,
                             **kwargs):
    """Rewrite `simple_test` for default backend."""
    assert self.with_bbox, 'Bbox head must be implemented.'
    x = self.extract_feat(img)
    if proposals is None:
        proposals, _ = self.rpn_head.simple_test_rpn(x, img_metas)
    return self.roi_head.simple_test(x, proposals, img_metas, rescale=False)
