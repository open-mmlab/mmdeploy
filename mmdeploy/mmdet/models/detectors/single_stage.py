from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.SingleStageDetector.simple_test')
def simple_test_of_single_stage(ctx, self, img, img_metas, **kwargs):
    """Rewrite `simple_test` for default backend."""
    feat = self.extract_feat(img)
    return self.bbox_head.simple_test(feat, img_metas, **kwargs)
