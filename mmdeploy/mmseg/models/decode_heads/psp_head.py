from mmseg.ops import resize

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.decode_heads.psp_head.PPM.forward')
def forward_of_ppm(ctx, self, x):
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
