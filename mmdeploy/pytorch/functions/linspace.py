# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from torch.types import Number

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.linspace')
def linspace__default(start: Number,
                      end: Number,
                      steps: Optional[int] = None,
                      **kwargs):
    """Rewrite `linspace` for default backend."""
    steps = 100 if steps is None else steps
    if steps >= 1:
        step = (end - start) * 1. / (steps - 1)
    else:
        step = 1
    output = torch.arange(start, end + 1, step, **kwargs)[:steps]
    return output
