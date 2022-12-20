# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER, SYMBOLIC_REWRITER
from mmdeploy.utils import IR


@FUNCTION_REWRITER.register_rewriter(
    'mmcv.ops.modulated_deform_conv.modulated_deform_conv2d',
    ir=IR.TORCHSCRIPT)
def modulated_deform_conv__torchscript(input, offset, mask, weight, bias,
                                       stride, padding, dilation, groups,
                                       deform_groups):
    """rewriter for the custom torchscript mdcn op."""
    from mmdeploy.backend.torchscript import get_ops_path, ops_available
    assert ops_available(), 'torchscript custom ops is required.'
    torch.ops.load_library(get_ops_path())
    from torch.nn.modules.utils import _pair
    kernel_h, kernel_w = weight.shape[-2:]
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    with_bias = bias is not None
    if not with_bias:
        bias = input.new_empty(0)
    return torch.ops.mmdeploy.modulated_deform_conv(
        input, weight, bias, offset, mask, kernel_h, kernel_w, stride[1],
        stride[0], padding[1], padding[0], dilation[1], dilation[0], groups,
        deform_groups, with_bias)


@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dFunction')
def modulated_deform_conv_default(g, input, offset, mask, weight, bias, stride,
                                  padding, dilation, groups, deform_groups):
    """Rewrite mdcn symbolic function for all backend."""
    input_tensors = [input, offset, mask, weight]
    if bias is not None:
        input_tensors.append(bias)
    return g.op(
        'mmdeploy::MMCVModulatedDeformConv2d',
        *input_tensors,
        stride_i=stride,
        padding_i=padding,
        dilation_i=dilation,
        groups_i=groups,
        deform_groups_i=deform_groups)
