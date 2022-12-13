# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import SYMBOLIC_REWRITER


# from torch import fake_quantize_per_tensor_affine
# from torch import fake_quantize_per_channel_affine
@SYMBOLIC_REWRITER.register_symbolic(
    'adaptive_avg_pool2d', is_pytorch=True, backend='ncnn')
def adaptive_avg_pool2d__ncnn(g, x, output_size):
    """Register ncnn symbolic function for `adaptive_avg_pool2d`.

    Align symbolic of adaptive_avg_pool2d in ncnn.
    """
    return g.op('mmdeploy::AdaptiveAvgPool2d', x, output_size)


# @SYMBOLIC_REWRITER.register_symbolic(
#     'fake_quantize_per_tensor_affine', is_pytorch=True, backend='snpe')
# def fake_quantize_per_tensor_affine__default(ctx, g, x, scale, zero_point, quant_min, quant_max ):
#     """Register ncnn symbolic function for `adaptive_avg_pool2d`.

#     Align symbolic of adaptive_avg_pool2d in ncnn.
#     """
#     return g.op('mmdeploy::FixedPerTensorAffine', x,  scale, zero_point, quant_min, quant_max)

# @SYMBOLIC_REWRITER.register_symbolic(
#     'fake_quantize_per_channel_affine', is_pytorch=True, backend='snpe')
# def fake_quantize_per_tensor_affine__default(ctx, g, x, scale, zero_point, quant_min, quant_max ):
#     """Register ncnn symbolic function for `adaptive_avg_pool2d`.

#     Align symbolic of adaptive_avg_pool2d in ncnn.
#     """
#     return g.op('mmdeploy::FixedPerChannelAffine', x,  scale, zero_point, quant_min, quant_max)

from torch.onnx import register_custom_op_symbolic

# Register symbolic op for torch.quantize_function op.


def _fake_quantize_learnable_per_tensor_affine(g, x, scale, zero_point,
                                               quant_min, quant_max,
                                               grad_factor):
    return g.op('::LearnablePerTensorAffine', x, scale, zero_point, quant_min,
                quant_max)


register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine',
                            _fake_quantize_learnable_per_tensor_affine, 11)


def fake_quantize_per_channel_affine(g, x, scale, zero_point, ch_axis,
                                     quant_min, quant_max):
    return g.op('mmdeploy::FixedPerChannelAffine', x, scale, zero_point,
                ch_axis, quant_min, quant_max)


register_custom_op_symbolic('::fake_quantize_per_channel_affine',
                            fake_quantize_per_channel_affine, 11)


def fake_quantize_per_tensor_affine(g, x, scale, zero_point, quant_min,
                                    quant_max):
    return g.op('mmdeploy::FixedPerTensorAffine', x, scale, zero_point,
                quant_min, quant_max)


register_custom_op_symbolic('::fake_quantize_per_tensor_affine',
                            fake_quantize_per_tensor_affine, 11)
