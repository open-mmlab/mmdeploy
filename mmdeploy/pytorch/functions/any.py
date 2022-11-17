# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.Tensor.any')
@FUNCTION_REWRITER.register_rewriter(func_name='torch.any')
def any__default(ctx, input, *args, **kwargs) -> torch.Tensor:
    """Rewrite `any` for ONNX."""
    if len(args) == 0 and kwargs == {}:
        return (input != 0).float().sum() > 0

    keepdim = False
    if len(args) == 2:
        keepdim = args[1]
    if 'keepdim' in kwargs:
        keepdim = kwargs['keepdim']
    dim = None
    if len(args) > 0 and isinstance(args[0], int):
        dim = args[0]
    if 'dim' in kwargs:
        dim = kwargs['dim']
    return (input != 0).sum(dim, keepdim=keepdim) > 0
