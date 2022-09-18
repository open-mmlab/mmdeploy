# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.clip', backend=Backend.COREML.value)
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.clip', backend=Backend.COREML.value)
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.clamp', backend=Backend.COREML.value)
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.clamp', backend=Backend.COREML.value)
def clip__coreml(ctx, input, min=None, max=None, **kwargs) -> torch.Tensor:
    """Rewrite `clip` for coreml backend.

    Cast data type.
    """
    if min is not None and not isinstance(min, torch.Tensor):
        min = input.new_tensor(min)

    if max is not None and not isinstance(max, torch.Tensor):
        max = input.new_tensor(max)

    return ctx.origin_func(input, min=min, max=max, **kwargs)
