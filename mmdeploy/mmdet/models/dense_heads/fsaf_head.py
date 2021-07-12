from mmdeploy.core import FUNCTION_REWRITER, mark


@FUNCTION_REWRITER.register_rewriter('mmdet.models.FSAFHead.forward')
@mark('rpn_forward', outputs=['cls_score', 'bbox_pred'])
def forward_of_fsaf_head(ctx, *args):
    return ctx.origin_func(*args)
