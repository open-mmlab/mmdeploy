# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.utils.UpConvBlock.forward')
def up_conv_block__forward(ctx, self, skip, x):
    """Rewrite `forward` for default backend.

    To support dynamic shape for UNet backbone,
    upsample feature maps with `size` instead of `scale_factor`

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        skip (Tensor): Skip branch feature.
        x (Tensor): Input feature to be upsampled.

    Returns:
        Tensor: Upsampled output feature map.
    """
    from mmcv.cnn import ConvModule

    # only valid when self.upsample is from build_upsample_layer
    if is_dynamic_shape(ctx.cfg) and not isinstance(self.upsample, ConvModule):
        # upsample with `size` instead of `scale_factor`
        from mmseg.ops import Upsample
        for c in self.upsample.interp_upsample:
            if isinstance(c, Upsample):
                c.size = skip.shape[-2:]
                c.scale_factor = None

    x = self.upsample(x)
    out = torch.cat([skip, x], dim=1)
    out = self.conv_block(out)
    return out
