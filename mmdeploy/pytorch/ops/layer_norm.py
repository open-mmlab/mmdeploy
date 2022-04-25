# Copyright (c) OpenMMLab. All rights reserved.
# Modified from:
# https://github.com/pytorch/pytorch/blob/9ade03959392e5a90b74261012de1d806cab2253/torch/onnx/symbolic_opset9.py

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import _unimplemented, parse_args

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend, get_backend


def mul(g, self, other):
    return g.op('Mul', self, other)


def add(g, self, other, alpha=None):
    if sym_help._is_value(self) and sym_help._is_tensor_list(self):
        return sym_help._onnx_opset_unsupported_detailed(
            'Add', 9, 11, 'Add between list of tensors not supported')

    # default alpha arg is to allow no-alpha add
    if alpha and sym_help._scalar(sym_help._maybe_get_scalar(alpha)) != 1:
        return _unimplemented('add', 'alpha != 1')
    return g.op('Add', self, other)


def sub(g, self, other, alpha=None):
    # default alpha arg is to allow no-alpha sub
    if alpha and sym_help._scalar(sym_help._maybe_get_scalar(alpha)) != 1:
        return _unimplemented('sub', 'alpha != 1')
    return g.op('Sub', self, other)


def sqrt(g, self):
    return g.op('Sqrt', self)


@parse_args('v', 'is', 'v', 'v', 'f', 'i')
def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
    if sym_help.is_caffe2_aten_fallback():
        return g.at(
            'layer_norm',
            input,
            weight,
            bias,
            normalized_shape_i=normalized_shape,
            eps_f=eps,
            cudnn_enable_i=cudnn_enable)

    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    two_cst = sym_help._generate_wrapped_number(g, 2.)
    eps_cst = sym_help._generate_wrapped_number(g, eps)

    mean = g.op('ReduceMean', input, axes_i=axes)
    numerator = sub(g, input, mean)
    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator
    # in the layer_norm formula
    variance = g.op('ReduceMean', pow(g, numerator, two_cst), axes_i=axes)
    denominator = sqrt(g, add(g, variance, eps_cst))

    layer_norm = g.op('Div', numerator, denominator)

    if not (weight is None or sym_help._is_none(weight)):
        layer_norm = mul(g, layer_norm, weight)
    if not (bias is None or sym_help._is_none(bias)):
        layer_norm = add(g, layer_norm, bias)

    return layer_norm


@parse_args('v', 'is', 'v', 'v', 'f', 'i')
def layer_norm_ncnn(g, input, normalized_shape, weight, bias, eps,
                    cudnn_enable):
    """Symbolic function for `layer_norm`.

    PyTorch does not support export layer_norm to ONNX by default. We add the
    support here. `layer_norm` will be exported as ONNX node
    'mmdeploy::layer_norm'
    """
    weight.setDebugName('layernorm_weight')
    bias.setDebugName('layernorm_bias')
    return g.op(
        'mmdeploy::LayerNorm', input, weight, bias, affine_i=1, epsilon_f=eps)


@SYMBOLIC_REWRITER.register_symbolic('layer_norm', is_pytorch=True)
def layer_norm__default(ctx, *args):
    """Register default symbolic function for `layer_norm`.

    Add support to layer_norm to ONNX.
    """
    backend = get_backend(ctx.cfg)
    if backend == Backend.NCNN:
        return layer_norm_ncnn(*args)
    else:
        return layer_norm(*args)
