# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.heads.MultiLabelClsHead.post_process')
def multi_label_cls_head__post_process(ctx, self, pred, **kwargs):
    """Rewrite `post_process` of MultiLabelClsHead for default backend.

    Rewrite this function to directly return pred.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        pred (Tensor): Predict result of model.

    Returns:
        pred (Tensor): Result of MultiLabelClsHead. The tensor
            shape (batch_size,num_classes).
    """
    return pred
