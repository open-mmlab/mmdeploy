from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textdet.SingleStageTextDetector.simple_test')
def simple_test_of_single_stage_text_detector(ctx,
                                              self,
                                              img,
                                              img_metas,
                                              rescale=False,
                                              **kwargs):
    """Rewrite `simple_test` for default backend."""
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    return outs
