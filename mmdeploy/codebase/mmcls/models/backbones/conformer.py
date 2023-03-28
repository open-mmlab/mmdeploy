# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.backbones.conformer.FCUUp.forward',
    backend=Backend.TENSORRT.value)
def fcuup__forward__tensorrt(self, x, H, W):
    """Rewrite `forward` of FCUUp used in conformer.

    FCUUp uses nearest interpolate while trt interpolate default
    is bilinear. Set interpolate mode explicitly.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (FCUUp): The instance of the class InvertedResidual.
        x (Tensor): Input features of shape (N, Cin, H, W).
        H (int): Feature map height
        W (int): Feature map width
    Returns:
        out (Tensor): A feature map output from FCUUp. The tensor
        shape (N, Cout, H * self.up_stride, W * self.up_stride).
    """
    B, _, C = x.shape
    # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
    if self.with_cls_token:
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
    else:
        x_r = x.transpose(1, 2).reshape(B, C, H, W)

    x_r = self.act(self.bn(self.conv_project(x_r)))

    return F.interpolate(
        x_r, size=(H * self.up_stride, W * self.up_stride), mode='nearest')
