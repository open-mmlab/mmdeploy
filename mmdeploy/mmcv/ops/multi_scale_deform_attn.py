# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import SYMBOLIC_REWRITER


@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction')
def ms_deform_attn_default(
    g,
    value,
    value_spatial_shapes,
    value_level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step=64,
):
    """Rewrite msda symbolic function for all backend."""
    return g.op(
        'mmdeploy::MMCVMultiScaleDeformableAttention',
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step_i=im2col_step,
    )
