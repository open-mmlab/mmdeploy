# Copyright (c) OpenMMLab. All rights reserved.
# Modified from:
# https://github.com/pytorch/pytorch/blob/9ade03959392e5a90b74261012de1d806cab2253/torch/onnx/symbolic_opset9.py

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend


@SYMBOLIC_REWRITER.register_symbolic(
    'linear',
    is_pytorch=True,
    arg_descriptors=['v', 'v', 'v', 'f', 'f', 'i', 'i'],
    backend=Backend.NCNN.value)
def linear__ncnn(ctx, g, input, weight, bias):
    """Support export linear This rewrite enable export Gemm."""
    return g.op(
        'mmdeploy::Gemm',
        input,
        weight,
        bias,
        alpha_f=1.0,
        beta_f=1.0,
        transA_i=0,
        transB_i=1)
