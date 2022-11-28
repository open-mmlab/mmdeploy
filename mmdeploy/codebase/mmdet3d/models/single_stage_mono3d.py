# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


def __forward_impl(ctx, self, img, cam2img, cam2img_inverse, img_metas,
                   **kwargs):
    """Rewrite `forward` function for SingleStageMono3DDetector.

    Support both dynamic and static export to onnx.
    """
    assert isinstance(img, torch.Tensor)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    img_shape = torch._shape_as_tensor(img)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    img_metas[0]['img_shape'] = img_shape
    return self.simple_test(
        img, cam2img, cam2img_inverse, img_metas, rescale=True)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.single_stage_mono3d.SingleStageMono3DDetector.'
    'forward')
def singlestagemono3d__forward(ctx,
                               self,
                               img,
                               cam2img,
                               cam2img_inverse,
                               img_metas=None,
                               return_loss=False,
                               **kwargs):
    """Rewrite this function to run the model directly."""
    if img_metas is None:
        img_metas = [{}]
    else:
        assert len(img_metas) == 1, 'do not support aug_test'
        img_metas = img_metas[0]

    if isinstance(img, list):
        img = img[0]

    return __forward_impl(
        ctx,
        self,
        img,
        cam2img,
        cam2img_inverse,
        img_metas=img_metas,
        **kwargs)
