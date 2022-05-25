# Copyright (c) OpenMMLab. All rights reserved.
import mmocr.utils as utils
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.BaseRecognizer.forward')
def base_recognizer__forward(ctx, self, img, img_metas=None, *args, **kwargs):
    """Rewrite `forward` of BaseRecognizer for default backend.

    Rewrite this function to:
    1. Create img_metas for exporting model to onnx.
    2. Call `simple_test` directly to skip `aug_test`.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class BaseRecognizer.
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        img_metas (Optional[list[dict]]): A list of image info dict where each
            dict has: 'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys, see
            :class:`mmdet.datasets.pipelines.Collect`.
        return_loss (bool): Whether compute and return loss. Used during
            training.

    Returns:
        out_dec (Tensor): A feature map output from a decoder. The tensor shape
            (N, H, W).
    """
    if img_metas is None:
        img_metas = [{}]
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    if utils.is_type_list(img_metas, list):
        img_metas = img_metas[0]
    assert utils.is_type_list(img_metas, dict)
    if utils.is_type_list(img, torch.Tensor):
        img = img[0]
    assert isinstance(img, torch.Tensor)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    img_shape = torch._shape_as_tensor(img)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    img_metas[0]['img_shape'] = img_shape
    return self.simple_test(img, img_metas, **kwargs)
