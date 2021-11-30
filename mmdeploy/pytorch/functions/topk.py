# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.topk', backend='default')
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.topk', backend='default')
def topk__dynamic(ctx,
                  input: torch.Tensor,
                  k: int,
                  dim: Optional[int] = None,
                  largest: bool = True,
                  sorted: bool = True):
    """Rewrite `topk` for default backend.

    Cast k to tensor and makesure k is smaller than input.shape[dim].
    """

    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k, device=input.device, dtype=torch.long)
    # Always keep topk op for dynamic input
    if isinstance(size, torch.Tensor):
        size = size.to(input.device)
    k = torch.where(k < size, k, size)
    return ctx.origin_func(input, k, dim=dim, largest=largest, sorted=sorted)


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.topk', backend='tensorrt')
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.topk', backend='tensorrt')
def topk__tensorrt(ctx,
                   input: torch.Tensor,
                   k: int,
                   dim: Optional[int] = None,
                   largest: bool = True,
                   sorted: bool = True):
    """Rewrite `topk` for TensorRT backend.

    TensorRT does not support topk with dynamic k. This function cast k to
    constant integer.
    """

    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if k > size:
        k = size
    if not isinstance(k, int):
        k = int(k)
    return ctx.origin_func(input, k, dim=dim, largest=largest, sorted=sorted)
