from typing import Tuple, Union

import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


def compute_padding(input_size, kernel_size, stride, dilation):
    """Compute padding 
    """
    
    input_h, input_w = input_size
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    output_h = math.ceil(input_h / stride_h)
    output_w = math.ceil(input_w / stride_w)
    pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - input_h, 0)
    pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - input_w, 0)
    if pad_w > 0 or pad_h > 0:
        padded = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
    else:
        padded = None
    return padded


class AdaptivePadOp(torch.autograd.Function):
    """AdaptivePadOp
    """
    
    @staticmethod
    def forward(ctx, x, kernel, stride, dilation):
        padded = compute_padding(x.shape[2:], kernel, stride, dilation)
        if padded is not None:
            x = F.pad(x, padded)
        return x
    
    @staticmethod
    def symbolic(g, x, kernel, stride, dilation):
        padded = compute_padding(x.type().sizes()[2:], kernel, stride, dilation)
        if padded is None:
            return g.op("Identity", x)
        padded = g.op("Constant", value_t=torch.tensor(padded, dtype=torch.int64))
        constant_value = g.op("Constant", value_t=torch.tensor(0, dtype=torch.float32))
        return g.op(
            "Pad",
            x,
            padded,
            constant_value,
            mode_s="constant",
            outputs=1
        )


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.cnn.bricks.conv2d_adaptive_padding.Conv2dAdaptivePadding.forward',
    backend=Backend.TENSORRT.value)
def conv2d_adaptive_padding__forward__tensorrt(ctx, self, x):
    """Implementation of 2D Convolution in tensorflow with `padding` as "same",
    which applies padding to input (if needed) so that the input image gets fully 
    covered by filter and stride you specify. For stride 1, this will ensure that the
    output image size is the same as input. For stride 2, output image size will be half.
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): size of convolution kernel
        stride (int or tuple, optional): Stride of convolution. Default 1
        padding (int or tuple, optional): Zero padding added to both sizes of 
            the input. Default: 0
        dilation (int or tuple, optional): spacing between kernel elements.
            Default: 1
        groups (int, optional): Number of blocked connections from input channels to
            output channels. Default: 1
        bias (bool, optional): If ``True``, add a learnable bias to the output.
            Default: ``True``
    """
    x = AdaptivePadOp.apply(x, self.weight.shape[2:], self.stride, self.dilation)
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
