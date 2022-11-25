# Copyright (c) OpenMMLab. All rights reserved.
# copy from https://github.com/pytorch/pytorch/blob/v1.12.1/torch/onnx/symbolic_opset9.py # noqa: E501
import torch
from torch.onnx import symbolic_helper

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend


@SYMBOLIC_REWRITER.register_symbolic(
    'linspace', is_pytorch=True, backend=Backend.DEFAULT.value)
def _linspace__default(ctx, g, start, end, steps, dtype, layout, device,
                       pin_memory):
    """rewriter of linspace.

    fix symbolic error of torch<=1.11
    """
    from torch.onnx.symbolic_opset9 import add, div, mul, sub
    range_tensor = symbolic_helper._arange_helper(g, steps, None)
    step = div(
        g,
        sub(g, end, start),
        sub(g, steps,
            g.op('Constant', value_t=torch.tensor(1, dtype=torch.int64))),
    )
    return add(g, mul(g, range_tensor, step), start)
