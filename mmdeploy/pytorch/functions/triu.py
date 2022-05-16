# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.triu')
def triu(ctx,
         input: torch.Tensor,
         diagonal: int = 0,
         *args,
         **kwargs) -> torch.Tensor:
    """Rewrite `triu` for exporting model to ONNX."""
    assert len(input.shape) >= 2
    height, width = input.shape[-2:]
    arange = torch.arange(width, device=input.device)
    mask = arange.expand(height, width)
    arange = arange.unsqueeze(-1)
    # statically conversion
    if height <= width:
        arange = arange[:height]
    else:
        arange = torch.cat(
            [arange, arange.new_zeros(height - width, 1) + width])
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    mask.expand(input.shape)
    return input * mask
