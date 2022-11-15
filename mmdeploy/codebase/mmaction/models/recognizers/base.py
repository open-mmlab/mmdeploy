# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmaction.models.recognizers.BaseRecognizer.forward')
def base_recognizer__forward(ctx,
                             self,
                             imgs: Tensor,
                             label=None,
                             return_loss=False,
                             **kwargs):
    """Rewrite `forward` of Recognizer2D for default backend."""

    assert kwargs.get('gradcam', False) is False
    assert return_loss is False
    return self.forward_test(imgs, **kwargs)
