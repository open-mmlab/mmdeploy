# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from:
# https://github.com/pytorch/pytorch/blob/9ade03959392e5a90b74261012de1d806cab2253/torch/onnx/symbolic_opset9.py
from mmdeploy.core import SYMBOLIC_REWRITER


@SYMBOLIC_REWRITER.register_symbolic(
    'hardsigmoid', is_pytorch=True, arg_descriptors=['v'])
def hardsigmoid__default(g, self):
    """Support export hardsigmoid This rewrite enable export hardsigmoid in
    torch<=1.8.2."""
    return g.op('HardSigmoid', self, alpha_f=1 / 6)
