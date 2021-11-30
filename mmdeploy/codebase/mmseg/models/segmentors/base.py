# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.BaseSegmentor.forward')
def base_segmentor__forward(ctx, self, img, img_metas=None, **kwargs):
    """Rewrite `forward` for default backend.

    Support configured dynamic/static shape for model input.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).
        img_metas (List[dict]): List of dicts containing image's meta
            information such as `img_shape`.

    Returns:
        torch.Tensor: Output segmentation map pf shape [N, 1, H, W].
    """
    if img_metas is None:
        img_metas = {}
    while isinstance(img_metas, list):
        img_metas = img_metas[0]

    if isinstance(img, list):
        img = torch.cat(img, 0)
    assert isinstance(img, torch.Tensor)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    img_shape = img.shape[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    img_metas['img_shape'] = img_shape
    return self.simple_test(img, img_metas, **kwargs)
