# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch.types import Number

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.linspace')
@FUNCTION_REWRITER.register_rewriter(func_name='torch.Tensor.linspace')
def linspace__onnx(ctx,
                   start: Number,
                   end: Number,
                   steps: Optional[int] = None,
                   **kwargs):
    """Rewrite `linspace` for onnxruntime."""
    steps = 100 if steps is None else steps
    if steps == 1:
        output = torch.arange(start, end + 1, **kwargs)[:steps]
    else:
        output = torch.arange(start, end + 1, (end - start) * 1. / (steps - 1),
                              **kwargs)[:steps]
    return output
