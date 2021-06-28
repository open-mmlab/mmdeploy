from mmdeploy.utils import FUNCTION_REWRITERS, mark


@FUNCTION_REWRITERS.register_rewriter('mmdet.models.FSAFHead.forward')
@mark('rpn_forward')
def fsaf_head_forward(rewriter, *args):
    return rewriter.origin_func(*args)
