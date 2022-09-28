# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend, is_dynamic_shape


def compute_padding(input_size, kernel_size, stride, dilation):
    """Compute padding."""

    input_h, input_w = input_size
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    output_h = math.ceil(input_h / stride_h)
    output_w = math.ceil(input_w / stride_w)
    pad_h = max(
        (output_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - input_h,
        0)
    pad_w = max(
        (output_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - input_w,
        0)
    if pad_w > 0 or pad_h > 0:
        padded = [
            pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
        ]
    else:
        padded = None
    return padded


class AdaptivePadOp(torch.autograd.Function):
    """AdaptivePadOp."""

    @staticmethod
    def forward(ctx, x, kernel, stride, dilation):
        padded = compute_padding(x.shape[2:], kernel, stride, dilation)
        if padded is not None:
            x = F.pad(x, padded)
        return x

    @staticmethod
    def symbolic(g, x, kernel, stride, dilation):
        padded = compute_padding(x.type().sizes()[2:], kernel, stride,
                                 dilation)
        if padded is None:
            return g.op('Identity', x)
        padded = g.op(
            'Constant', value_t=torch.tensor(padded, dtype=torch.int64))
        constant_value = g.op(
            'Constant', value_t=torch.tensor(0, dtype=torch.float32))
        return g.op(
            'Pad', x, padded, constant_value, mode_s='constant', outputs=1)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.cnn.bricks.conv2d_adaptive_padding. \
        Conv2dAdaptivePadding.forward',
    backend=Backend.TENSORRT.value)
def conv2d_adaptive_padding__forward__tensorrt(ctx, self, x):
    """Rewrite `forward` of Conv2dAdaptivePadding used in EfficientNet for
    TensorRT backend. Main changes of this rewritten function is to separate
    the computation of padding and encapsulate it into another
    `torch.autograd.Function` so that the adaptive padding could be parsed as
    `Pad` ops in ONNX with the padding information computed in advance (Only
    for static shape configuration).

    Args:
        x (Tensor): Input tensor of Conv2dAdaptivePadding ops
    Returns:
        Tensor: forward result of 2D convolution after padding
    """

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    if not is_dynamic_flag:
        x = AdaptivePadOp.apply(x, self.weight.shape[2:], self.stride,
                                self.dilation)
    else:
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = (
            max((output_h - 1) * self.stride[0] +
                (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0))
        pad_w = (
            max((output_w - 1) * self.stride[1] +
                (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [
                pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2
            ])
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)
