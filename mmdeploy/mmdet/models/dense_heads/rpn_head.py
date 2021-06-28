from mmdeploy.utils import FUNCTION_REWRITERS, mark


@FUNCTION_REWRITERS.register_rewriter('mmdet.models.RPNHead.forward')
@mark('rpn_forward')
def rpn_head_forward(rewriter, self, feats):
    return rewriter.origin_func(self, feats)
