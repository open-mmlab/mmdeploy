# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape


@mark(
    'detector_forward', inputs=['input'], outputs=['dets', 'labels', 'masks'])
def __forward_impl(ctx, self, img, img_metas, **kwargs):
    """Rewrite and adding mark for `forward`.

    Encapsulate this function for rewriting `forward` of BaseDetector.
    1. Add mark for BaseDetector.
    2. Support both dynamic and static export to onnx.
    """
    assert isinstance(img, torch.Tensor)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    img_shape = torch._shape_as_tensor(img)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    img_metas[0]['img_shape'] = img_shape
    return self.simple_test(img, img_metas, **kwargs)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.base.BaseDetector.forward')
def base_detector__forward(ctx,
                           self,
                           img,
                           img_metas=None,
                           return_loss=False,
                           **kwargs):
    """Rewrite `forward` of BaseDetector for default backend.

    Rewrite this function to:
    1. Create img_metas for exporting model to onnx.
    2. Call `simple_test` directly to skip `aug_test`.
    3. Remove `return_loss` because deployment has no need for training
    functions.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class BaseDetector.
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        img_metas (Optional[list[dict]]): A list of image info dict where each
            dict has: 'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys, see
            :class:`mmdet.datasets.pipelines.Collect`.

    Returns:
        list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
    """
    if img_metas is None:
        img_metas = [{}]
    else:
        assert len(img_metas) == 1, 'do not support aug_test'
        img_metas = img_metas[0]

    if isinstance(img, list):
        img = img[0]

    return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)
