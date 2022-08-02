# Copyright (c) OpenMMLab. All rights reserved.
# modified from
# https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py
import sys

from torch.onnx.symbolic_helper import _slice_helper, parse_args

from mmdeploy.core import SYMBOLIC_REWRITER


@parse_args('v', 'is', 'is')
def roll(g, self, shifts, dims):
    """Symbolic function for `roll`."""
    assert len(shifts) == len(dims)

    result = self
    for i in range(len(shifts)):
        shapes = []
        shape = _slice_helper(
            g, result, axes=[dims[i]], starts=[-shifts[i]], ends=[sys.maxsize])
        shapes.append(shape)
        shape = _slice_helper(
            g, result, axes=[dims[i]], starts=[0], ends=[-shifts[i]])
        shapes.append(shape)
        result = g.op('Concat', *shapes, axis_i=dims[i])

    return result


@SYMBOLIC_REWRITER.register_symbolic('roll', is_pytorch=True)
def roll_default(ctx, g, self, shifts, dims):
    """Support export roll to ONNX with PyTorch version 1.10-."""
    return roll(g, self, shifts, dims)
