# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.triu')
def triu(ctx, input: torch.Tensor, diagonal: int = 0, *args, **kwargs):
    """Rewrite `linear` for NCNN backend.

    The broadcast rules are different between ncnn and PyTorch. This function
    add extra reshape and transpose to support linear operation of different
    input shape.
    """
    size = input.shape[0]
    arange = torch.arange(size, device=input.device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return input.masked_fill(mask == 0, 0)
