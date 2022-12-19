# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from:
# https://github.com/pytorch/pytorch/blob/9ade03959392e5a90b74261012de1d806cab2253/torch/onnx/symbolic_opset9.py

import torch
from torch.onnx.symbolic_helper import (_get_tensor_dim_size, _get_tensor_rank,
                                        _unimplemented, _unsqueeze_helper,
                                        parse_args)

from mmdeploy.core import SYMBOLIC_REWRITER


@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def instance_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    """Symbolic function for `instance_norm`."""
    channel_size = _get_tensor_dim_size(input, 1)
    if channel_size is not None:
        assert channel_size % num_groups == 0
    input_rank = _get_tensor_rank(input)
    if input_rank is None:
        return _unimplemented('group_norm', 'unknown input rank')
    # 0 in the shape list keeps dimension value unchanged.
    shape = [0, num_groups, -1]
    input_reshaped = g.op('Reshape', input,
                          g.op('Constant', value_t=torch.LongTensor(shape)))

    # C is always divisible by num_groups
    # Due to shape difference. we need to apply weight and bias after
    # instance norm computation and reshape
    weight_ = g.op(
        'Constant',
        value_t=torch.tensor(
            [1.] * num_groups).type('torch.' + input.type().scalarType() +
                                    'Tensor'))
    bias_ = g.op(
        'Constant',
        value_t=torch.tensor(
            [0.] * num_groups).type('torch.' + input.type().scalarType() +
                                    'Tensor'))

    norm_reshaped = g.op(
        'mmdeploy::TRTInstanceNormalization',
        input_reshaped,
        weight_,
        bias_,
        epsilon_f=eps)
    norm = g.op('Reshape', norm_reshaped, g.op('Shape', input))

    if weight is None or weight.node().mustBeNone():
        weight_value = torch.tensor(
            [1.]).type('torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op('Constant', value_t=weight_value)
    if bias is None or bias.node().mustBeNone():
        bias_value = torch.tensor(
            [0.]).type('torch.' + input.type().scalarType() + 'Tensor')
        bias = g.op('Constant', value_t=bias_value)

    # Norm has shape [N, C, *] so we reshape weight and bias to [C, *]
    axes = list(range(1, input_rank - 1))
    from torch.onnx.symbolic_opset9 import add, mul
    return add(g, mul(g, norm, _unsqueeze_helper(g, weight, axes)),
               _unsqueeze_helper(g, bias, axes))


@SYMBOLIC_REWRITER.register_symbolic(
    'group_norm', backend='tensorrt', is_pytorch=True)
def instance_norm__tensorrt(*args):
    """Register symbolic function for TensorRT backend.

    Notes:
        Instance normalization is implemented in group norm in pytorch.
    """
    return instance_norm(*args)
