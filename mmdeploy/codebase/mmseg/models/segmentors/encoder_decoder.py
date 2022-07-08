# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.EncoderDecoder.predict')
def encoder_decoder__simple_test(ctx, self, batch_inputs, batch_data_samples,
                                 **kwargs):
    """Rewrite `predict` for default backend.

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
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
    seg_logit = self.encode_decode(batch_inputs, batch_img_metas)
    seg_logit = F.softmax(seg_logit, dim=1)
    seg_pred = seg_logit.argmax(dim=1, keepdim=True)
    return seg_pred
