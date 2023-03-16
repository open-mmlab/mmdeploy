# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Sequence, Union

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.normalize', backend='ncnn')
def normalize__ncnn(input: torch.Tensor,
                    p: int = 2,
                    dim: int = 1,
                    eps: float = 1e-12,
                    *args,
                    **kwargs):
    """Rewrite `normalize` for ncnn backend.

    Make sure L2 norm on channel dim and be exported to ncnn correctly.
    """
    ctx = FUNCTION_REWRITER.get_context()
    if dim < 0:
        dim += input.ndim
    assert dim != 0, 'Should not normalize on batch index'
    origin_func = ctx.origin_func
    assert p == 2, 'only support L2 norm'
    assert input.ndim in [3, 4]
    assert input.shape[0] == 1, \
        f'only support batch size 1, but given {input.shape[0]}'
    if input.ndim == 3:
        output = origin_func(
            input.transpose(1, dim).unsqueeze(2), p=p, dim=1,
            eps=eps).squeeze(2).transpose(1, dim)
    else:
        # input.ndim == 4:
        if dim == 1:
            output = origin_func(input, p=p, dim=dim, eps=eps)
        else:
            output = origin_func(
                input.transpose(1, dim), p=p, dim=1,
                eps=eps).transpose(1, dim)
    return output


@FUNCTION_REWRITER.register_rewriter(func_name='torch.norm', backend='ncnn')
def norm__ncnn(input: torch.Tensor,
               p: Optional[Union[int, str]] = 'fro',
               dim: Optional[Union[int, Sequence]] = None,
               keepdim: Optional[bool] = False,
               out: Optional[torch.Tensor] = None,
               dtype: Optional[torch.dtype] = None):
    """Rewrite `torch.norm` for ncnn backend.

    Rewrite torch.norm when p is Frobenius norm to avoid FP16 exceed in ncnn
    Android platform.
    """
    ctx = FUNCTION_REWRITER.get_context()
    origin_func = ctx.origin_func
    if p == 'fro' and (isinstance(dim, int) or len(dim) == 1):
        # Substitute Frobenius norm with L2 norm.
        return origin_func(
            input, p=2, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
    else:
        return origin_func(
            input, p=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
