from mmdeploy.core import FUNCTION_REWRITER, mark


@FUNCTION_REWRITER.register_rewriter('mmdet.models.RPNHead.forward')
@mark(
    'rpn_forward', inputs='feats', outputs=['rpn_cls_score', 'rpn_bbox_pred'])
def forward_of_rpn_head(ctx, self, feats):
    return ctx.origin_func(self, feats)
