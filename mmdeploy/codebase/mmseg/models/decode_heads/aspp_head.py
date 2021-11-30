# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmseg.ops import resize

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.decode_heads.ASPPHead.forward')
def aspp_head__forward(ctx, self, inputs):
    """Rewrite `forward` for default backend.

    Support configured dynamic/static shape in resize op.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        inputs (list[Tensor]): List of multi-level img features.

    Returns:
        torch.Tensor: Output segmentation map.
    """
    x = self._transform_inputs(inputs)
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    size = x.shape[2:]
    if not is_dynamic_flag:
        size = [int(val) for val in size]

    aspp_outs = [
        resize(
            self.image_pool(x),
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
    ]
    aspp_outs.extend(self.aspp_modules(x))
    aspp_outs = torch.cat(aspp_outs, dim=1)
    output = self.bottleneck(aspp_outs)
    output = self.cls_seg(output)
    return output
