# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import get_root_logger


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
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#topKsetup
    MAX_TOPK_K = 3840
    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if k > size:
        k = size
    if not isinstance(k, int):
        k = int(k)
    if k > MAX_TOPK_K:
        logger = get_root_logger()
        logger.warning(
            f'Maximum K of TopK in TensorRT is {MAX_TOPK_K}, but given {k}.'
            f' Note that k will be set to {MAX_TOPK_K}.')
        k = MAX_TOPK_K

    return ctx.origin_func(input, k, dim=dim, largest=largest, sorted=sorted)
