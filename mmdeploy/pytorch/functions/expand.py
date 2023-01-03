# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.expand', backend='ncnn')
def expand__ncnn(self, *sizes) -> torch.Tensor:
    """Rewrite `expand` for NCNN backend.

    Do not expand on batch dim for tensor with ndim >= 3
    """
    ctx = FUNCTION_REWRITER.get_context()
    if self.ndim < 3 or sizes[0] not in [1, -1]:
        return ctx.origin_func(*sizes)
    return self
