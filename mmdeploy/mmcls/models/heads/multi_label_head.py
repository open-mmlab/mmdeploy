from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.heads.MultiLabelClsHead.post_process')
def post_process_of_multi_label_head(ctx, self, pred, **kwargs):
    """Rewrite `post_process` for default backend."""
    return pred
