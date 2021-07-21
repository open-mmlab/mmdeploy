from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='mmdet.models.RPN.simple_test')
def simple_test_of_rpn(ctx, self, img, img_metas, **kwargs):
    x = self.extract_feat(img)
    return self.rpn_head.simple_test_rpn(x, img_metas)
