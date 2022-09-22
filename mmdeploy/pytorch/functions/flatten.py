# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.flatten', backend=Backend.COREML.value)
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.flatten', backend=Backend.COREML.value)
def flatten__coreml(ctx, input, start_dim=0, end_dim=-1) -> torch.Tensor:
    """Rewrite `flatten` for coreml backend.

    Use reshape instead of flatten
    """
    shape = input.shape
    end_dim = end_dim if end_dim > 0 else len(shape) + end_dim
    shape1 = list(shape[:start_dim])
    shape3 = list(shape[end_dim + 1:])
    return input.reshape(shape1 + [-1] + shape3)
