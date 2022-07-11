# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmcls.models.classifiers.ImageClassifier.forward', backend='default')
@FUNCTION_REWRITER.register_rewriter(
    'mmcls.models.classifiers.BaseClassifier.forward', backend='default')
def base_classifier__forward(ctx, self, img, return_loss=False, **kwargs):
    """Rewrite `forward` of BaseClassifier for default backend.

    Rewrite this function to call simple_test function,
    ignore the return_loss parameter.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (List[Tensor]): The outer list indicates test-time
            augmentations and inner Tensor should have a shape NxCxHxW,
            which contains all images in the batch.

    Returns:
        result(Tensor): The result of classifier.The tensor
            shape (batch_size,num_classes).
    """
    result = self.simple_test(img, **kwargs)
    return result
