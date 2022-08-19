# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import SYMBOLIC_REWRITER


@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.deform_conv.DeformConv2dFunction')
def deform_conv__default(ctx,
                         g,
                         input,
                         offset,
                         weight,
                         stride,
                         padding,
                         dilation,
                         groups,
                         deform_groups,
                         bias=False,
                         im2col_step=32):
    """Rewrite symbolic function for default backend."""
    return g.op(
        'mmdeploy::MMCVDeformConv2d',
        input,
        offset,
        weight,
        stride_i=stride,
        padding_i=[p for pair in zip(padding, padding) for p in pair],
        dilation_i=dilation,
        groups_i=groups,
        deform_groups_i=deform_groups)


@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.deform_conv.DeformConv2dFunction', backend='openvino')
def deform_conv_openvino(ctx,
                         g,
                         input,
                         offset,
                         weight,
                         stride,
                         padding,
                         dilation,
                         groups,
                         deform_groups,
                         bias=False,
                         im2col_step=32):
    """Rewrite symbolic function for OpenVINO backend."""
    assert not bias, 'The "bias" parameter should be False.'
    assert groups == 1, 'The "groups" parameter should be 1.'
    kh, kw = weight.type().sizes()[2:]
    domain = 'org.openvinotoolkit'
    op_name = 'DeformableConv2D'
    return g.op(
        f'{domain}::{op_name}',
        input,
        offset,
        weight,
        strides_i=stride,
        pads_i=[p for pair in zip(padding, padding) for p in pair],
        dilations_i=dilation,
        groups_i=groups,
        deformable_groups_i=deform_groups,
        kernel_shape_i=[kh, kw])
