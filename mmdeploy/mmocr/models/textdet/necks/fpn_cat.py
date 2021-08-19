import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textdet.FPNC.forward', backend='tensorrt')
def forward_of_fpnc(ctx, self, inputs, **kwargs):
    """Replace nearest upsampling with bilinear upsampling."""

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
            laterals[i], size=prev_shape, mode='bilinear')
    # build outputs
    # part 1: from original levels
    outs = [
        self.smooth_convs[i](laterals[i]) for i in range(used_backbone_levels)
    ]

    for i, out in enumerate(outs):
        outs[i] = F.interpolate(
            outs[i], size=outs[0].shape[2:], mode='bilinear')
    out = torch.cat(outs, dim=1)

    if self.conv_after_concat:
        out = self.out_conv(out)

    return out
