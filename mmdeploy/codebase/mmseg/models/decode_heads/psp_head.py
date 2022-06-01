# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmseg.ops import resize

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import IR, get_root_logger, is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.decode_heads.psp_head.PPM.forward', ir=IR.ONNX)
def ppm__forward(ctx, self, x):
    """Rewrite `forward` for default backend.

    Support configured dynamic/static shape in resize op.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        x (Tensor): The transformed input feature.

    Returns:
        List[torch.Tensor]: Up-sampled segmentation maps of different
            scales.
    """
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    size = x.shape[2:]
    if not is_dynamic_flag:
        size = [int(val) for val in size]

    ppm_outs = []
    for ppm in self:
        if isinstance(ppm[0], nn.AdaptiveAvgPool2d) and \
                ppm[0].output_size != 1:
            if is_dynamic_flag:
                logger = get_root_logger()
                logger.warning('`AdaptiveAvgPool2d` would be '
                               'replaced to `AvgPool2d` explicitly')
            # replace AdaptiveAvgPool2d with AvgPool2d explicitly
            output_size = 2 * [ppm[0].output_size]
            k = [int(size[i] / output_size[i]) for i in range(0, len(size))]
            ppm[0] = nn.AvgPool2d(k, stride=k, padding=0, ceil_mode=False)
        ppm_out = ppm(x)
        upsampled_ppm_out = resize(
            ppm_out,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        ppm_outs.append(upsampled_ppm_out)
    return ppm_outs
