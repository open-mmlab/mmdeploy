# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils.constants import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.EncoderDecoder.simple_test')
def encoder_decoder__simple_test(ctx, self, img, img_meta, **kwargs):
    """Rewrite `simple_test` for default backend.

    Support configured dynamic/static shape for model input and return
    segmentation map as Tensor instead of numpy array.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).
        img_meta (dict): Dict containing image's meta information
            such as `img_shape`.

    Returns:
        torch.Tensor: Output segmentation map pf shape [N, 1, H, W].
    """
    seg_logit = self.encode_decode(img, img_meta)
    seg_logit = F.softmax(seg_logit, dim=1)
    seg_pred = seg_logit.argmax(dim=1, keepdim=True)
    return seg_pred


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.EncoderDecoder.simple_test',
    backend=Backend.RKNN.value)
def encoder_decoder__simple_test__rknn(ctx, self, img, img_meta, **kwargs):
    """Rewrite `simple_test` for RKNN backend.

    Early return to avoid argmax operator.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).
        img_meta (dict): Dict containing image's meta information
            such as `img_shape`.

    Returns:
        torch.Tensor: Output segmentation map pf shape [N, C, H, W].
    """
    seg_logit = self.encode_decode(img, img_meta)
    seg_logit = F.softmax(seg_logit, dim=1)
    return seg_logit
