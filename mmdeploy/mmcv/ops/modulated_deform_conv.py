# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import SYMBOLIC_REWRITER


@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dFunction')
def modulated_deform_conv_default(ctx, g, input, offset, mask, weight, bias,
                                  stride, padding, dilation, groups,
                                  deform_groups):
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
