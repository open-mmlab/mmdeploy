# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmaction.models.recognizers.Recognizer2D.forward_test')
def recognizer2d__forward_test(ctx, self, imgs: Tensor):
    """Rewrite `forward_test` of Recognizer2D for default backend."""

    if self.test_cfg.get('fcn_test', False):
        # If specified, spatially fully-convolutional testing is performed
        assert not self.feature_extraction
        assert self.with_cls_head
        return self._do_fcn_test(imgs)

    return self._do_test(imgs)
