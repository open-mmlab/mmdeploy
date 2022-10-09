# Copyright (c) OpenMMLab. All rights reserved.
# Modified from:
# https://github.com/pytorch/pytorch/blob/9ade03959392e5a90b74261012de1d806cab2253/torch/onnx/symbolic_opset9.py
from torch.onnx.symbolic_helper import parse_args

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend


@parse_args('v', 'v', 'f', 'f', 'i', 'i')
def linear_no_bias(g, input, weight):
    """Symbolic function for `linear` without bias.

    PyTorch `nn.Linear` will be exported as ONNX node 'Gemm'.
    """
    return g.op(
        'Gemm', input, weight, alpha_f=1.0, beta_f=1.0, transA_i=0, transB_i=1)


@parse_args('v', 'v', 'v', 'f', 'f', 'i', 'i')
def linear_normal(g, input, weight, bias):
    """Symbolic function for `linear`.

    PyTorch `nn.Linear` will be exported as ONNX node 'Gemm'.
    """
    return g.op(
        'Gemm',
        input,
        weight,
        bias,
        alpha_f=1.0,
        beta_f=1.0,
        transA_i=0,
        transB_i=1)


@SYMBOLIC_REWRITER.register_symbolic(
    'linear', is_pytorch=True, backend=Backend.NCNN.value)
def linear__ncnn(ctx, g, input, weight, bias):
    """Support export linear This rewrite enable export Gemm."""
    if bias is None:
        return linear_no_bias(g, input, weight)
    else:
        return linear_normal(g, input, weight, bias)
