from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.heads.ClsHead.post_process')
def post_process_of_cls_head(ctx, self, pred, **kwargs):
    return pred
