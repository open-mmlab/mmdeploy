# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import SYMBOLIC_REWRITER


@SYMBOLIC_REWRITER.register_symbolic(
    'fake_quantize_per_tensor_affine', is_pytorch=True)
def fake_quantize_per_tensor_affine__default(ctx, g, x, scale, zero_point,
                                             quant_min, quant_max):
    """Register ncnn symbolic function for `adaptive_avg_pool2d`.

    Align symbolic of adaptive_avg_pool2d in ncnn.
    """
    return g.op('mmdeploy::FixedPerTensorAffine', x, scale, zero_point,
                quant_min, quant_max)


@SYMBOLIC_REWRITER.register_symbolic(
    'fake_quantize_per_channel_affine', is_pytorch=True)
def fake_quantize_per_channel_affine__default(ctx, g, x, scale, zero_point,
                                              ch_axis, quant_min, quant_max):
    """Register ncnn symbolic function for `adaptive_avg_pool2d`.

    Align symbolic of adaptive_avg_pool2d in ncnn.
    """
    return g.op('mmdeploy::FixedPerChannelAffine', x, scale, zero_point,
                ch_axis, quant_min, quant_max)
