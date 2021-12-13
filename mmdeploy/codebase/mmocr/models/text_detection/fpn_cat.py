# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from packaging import version

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textdet.FPNC.forward', backend='tensorrt')
def fpnc__forward__tensorrt(ctx, self, inputs, **kwargs):
    """Rewrite `forward` of FPNC for tensorrt backend.

    Rewrite this function to replace nearest upsampling with bilinear
    upsampling. TensorRT-7 backend applies different nearest sampling strategy
    from pytorch, which heavily influenced the final performance.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class FPNC.
        inputs (Sequence[Tensor]): The feature maps for each scale level with
            shape (N, num_anchors * num_classes, H, W)

    Returns:
        outs (Tensor): A feature map output from FPNC. The tensor shape
            (N, C, H, W).
    """
    # TensorRT version 8+ matches the upsampling with pytorch
    import tensorrt as trt
    apply_rewrite = version.parse(trt.__version__) < version.parse('8')
    mode = 'bilinear' if apply_rewrite else 'nearest'

    assert len(inputs) == len(self.in_channels)
    # build laterals
    laterals = [
        lateral_conv(inputs[i])
        for i, lateral_conv in enumerate(self.lateral_convs)
    ]
    used_backbone_levels = len(laterals)
    # build top-down path
    for i in range(used_backbone_levels - 1, 0, -1):
        prev_shape = laterals[i - 1].shape[2:]
        laterals[i - 1] += F.interpolate(
            laterals[i], size=prev_shape, mode=mode)
    # build outputs
    # part 1: from original levels
    outs = [
        self.smooth_convs[i](laterals[i]) for i in range(used_backbone_levels)
    ]

    for i, out in enumerate(outs):
        outs[i] = F.interpolate(outs[i], size=outs[0].shape[2:], mode=mode)
    out = torch.cat(outs, dim=1)

    if self.conv_after_concat:
        out = self.out_conv(out)

    return out
