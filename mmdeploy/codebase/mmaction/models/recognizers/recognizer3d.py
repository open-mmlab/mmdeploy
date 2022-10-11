# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmaction.models.recognizers.Recognizer3D.forward_test')
def recognizer3d__forward_test(ctx, self, imgs: Tensor):
    """Rewrite `forward_test` of Recognizer3D for default backend."""
    return self._do_test(imgs)
