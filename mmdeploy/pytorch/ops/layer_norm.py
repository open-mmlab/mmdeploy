# Copyright (c) OpenMMLab. All rights reserved.
# Modified from:
# https://github.com/pytorch/pytorch/blob/9ade03959392e5a90b74261012de1d806cab2253/torch/onnx/symbolic_opset9.py

from torch.onnx.symbolic_helper import parse_args

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend


@parse_args('v', 'is', 'v', 'v', 'f', 'i')
def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
    """Symbolic function for `layer_norm`.

    PyTorch does not support export layer_norm to ONNX by default. We add the
    support here. `layer_norm` will be exported as ONNX node
    'mmdeploy::layer_norm'
    """
    weight.setDebugName('layernorm_weight')
    bias.setDebugName('layernorm_bias')
    return g.op(
        'mmdeploy::LayerNorm', input, weight, bias, affine_i=1, epsilon_f=eps)


@SYMBOLIC_REWRITER.register_symbolic(
    'layer_norm', is_pytorch=True, backend=Backend.NCNN.value)
def layer_norm__ncnn(ctx, *args):
    """Register default symbolic function for `layer_norm`.

    Add support to layer_norm to ONNX.
    """
    return layer_norm(*args)
