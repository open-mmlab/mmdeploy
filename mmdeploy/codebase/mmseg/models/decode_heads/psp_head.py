# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.ops import resize

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.decode_heads.psp_head.PPM.forward')
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
        ppm_out = ppm(x)
        upsampled_ppm_out = resize(
            ppm_out,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        ppm_outs.append(upsampled_ppm_out)
    return ppm_outs
