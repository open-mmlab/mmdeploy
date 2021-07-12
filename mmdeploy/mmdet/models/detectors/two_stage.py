from mmdeploy.core import FUNCTION_REWRITER, mark


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.TwoStageDetector.extract_feat')
@mark('extract_feat', inputs='img', outputs='feat')
def extract_feat_of_two_stage(ctx, self, img):
    return ctx.origin_func(self, img)
