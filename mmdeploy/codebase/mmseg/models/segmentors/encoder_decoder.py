# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.EncoderDecoder.predict')
def encoder_decoder__predict(self, inputs, data_samples, **kwargs):
    """Rewrite `predict` for default backend.

    1. only support mode=`whole` inference
    2. skip calling self.postprocess_result

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        inputs (Tensor): Inputs with shape (N, C, H, W).
        data_samples (SampleList): The seg data samples.

    Returns:
        torch.Tensor: Output segmentation logits of shape [N, C, H, W].
    """
    batch_img_metas = []
    for data_sample in data_samples:
        batch_img_metas.append(data_sample.metainfo)
    x = self.extract_feat(inputs)
    seg_logit = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
    return seg_logit
