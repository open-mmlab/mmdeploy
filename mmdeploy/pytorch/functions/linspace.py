# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch.types import Number

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.linspace')
def linspace__default(start: Number, end: Number, steps: int = None, **kwargs):
    """Rewrite `linspace` for onnxruntime."""
    steps = 100 if steps is None else steps
    dtype = kwargs.pop('dtype', torch.float32)
    dtype = dtype if dtype else torch.float32
    if steps == 1:
        output = torch.arange(start, end + 1, dtype=dtype, **kwargs)[:steps]
    else:
        output = torch.arange(
            start, end + 1, (end - start) / (steps - 1), dtype=dtype,
            **kwargs)[:steps]
    return output
