# Copyright (c) OpenMMLab. All rights reserved.
# Modified from:
# https://github.com/pytorch/pytorch/blob/9ade03959392e5a90b74261012de1d806cab2253/torch/onnx/symbolic_opset9.py
import torch
from torch.onnx.symbolic_helper import parse_args

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend


@SYMBOLIC_REWRITER.register_symbolic(
    'layer_norm',
    is_pytorch=True,
    arg_descriptors=['v', 'is', 'v', 'v', 'f', 'i'])
def layer_norm__default(ctx, g, input, normalized_shape, weight, bias, eps,
                        cudnn_enable):
    """Symbolic function for `layer_norm`

    Layer norm with torch<=1.12 might lead to wrong output shapes. Add
    keepdims=1 to each ReduceMean node to correct the shape.
    """
    import torch.onnx.symbolic_helper as sym_help
    from torch.onnx.symbolic_opset9 import add, mul, pow, sqrt, sub

    axes = [-i for i in range(len(normalized_shape), 0, -1)]

    two_cst = sym_help._generate_wrapped_number(g, 2.)
    eps_cst = sym_help._generate_wrapped_number(g, eps)

    mean = g.op('ReduceMean', input, axes_i=axes, keepdims_i=1)
    numerator = sub(g, input, mean)
    # variance = e((x - e(x))^2), and (x - e(x)) is the numerator in the
    # layer_norm formula
    variance = g.op(
        'ReduceMean', pow(g, numerator, two_cst), axes_i=axes, keepdims_i=1)
    denominator = sqrt(g, add(g, variance, eps_cst))

    layer_norm = g.op('Div', numerator, denominator)

    if not (weight is None or sym_help._is_none(weight)):
        layer_norm = mul(g, layer_norm, weight)
    if not (bias is None or sym_help._is_none(bias)):
        layer_norm = add(g, layer_norm, bias)

    return layer_norm


@parse_args('v', 'is', 'v', 'v', 'f', 'i')
def _layer_norm_ncnn(g, input, normalized_shape, weight, bias, eps,
                     cudnn_enable):
    """Symbolic function for `layer_norm`.

    PyTorch does not support export layer_norm to ONNX by default. We add the
    support here. `layer_norm` will be exported as ONNX node
    'mmdeploy::layer_norm'
    """
    weight.setDebugName('layernorm_weight')
    bias.setDebugName('layernorm_bias')
    if isinstance(normalized_shape,
                  list) and len(normalized_shape) == 3 and normalized_shape[
                      1] == 1 and normalized_shape[2] == 1:
        """ncnn now has supported the layernorm which is used in NLP field
        instead of CV, there are some differences between them, so this is a
        special case could be replace by reshaping."""
        [c, h, w] = normalized_shape
        ori_shape = g.op('Constant', value_t=torch.LongTensor([1, c, h, w]))
        shape = g.op('Constant', value_t=torch.LongTensor([1, h * w, c]))
        input = g.op('Reshape', input, shape)
        input = g.op(
            'mmdeploy::LayerNorm',
            input,
            weight,
            bias,
            affine_i=1,
            epsilon_f=eps)
        input = g.op('Reshape', input, ori_shape)
        return input

    return g.op(
        'mmdeploy::LayerNorm', input, weight, bias, affine_i=1, epsilon_f=eps)


@SYMBOLIC_REWRITER.register_symbolic(
    'layer_norm', is_pytorch=True, backend=Backend.NCNN.value)
def layer_norm__ncnn(ctx, *args):
    """Register default symbolic function for `layer_norm`.

    Add support to layer_norm to ONNX.
    """
    return _layer_norm_ncnn(*args)
