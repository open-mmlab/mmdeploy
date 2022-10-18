# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.fcos_mono3d.FCOSMono3D.simple_test')
def fcosmono3d__simple_test(ctx,
                            self,
                            img,
                            cam2img,
                            cam2img_inverse,
                            img_metas,
                            rescale=False):
    """Test function without augmentaiton. Rewrite this func to remove model
    post process.

    Args:
        img (torch.Tensor): Input image tensor in shape (N, C, H, W).
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].
        cam2img_inverse (torch.Tensor): The inverse of camera intrinsic matrix.
        img_metas (list[dict]): Contain img meta info.

    Returns:
        List: Result of model.
    """
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    outs = self.bbox_head.get_bboxes(
        *outs, cam2img, cam2img_inverse, img_metas, rescale=rescale)

    return outs
