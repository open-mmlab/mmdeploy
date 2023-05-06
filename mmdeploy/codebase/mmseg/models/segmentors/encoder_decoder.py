# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import get_codebase_config


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
        torch.Tensor: Output segmentation map pf shape [N, 1, H, W].
    """
    batch_img_metas = []
    for data_sample in data_samples:
        batch_img_metas.append(data_sample.metainfo)
    x = self.extract_feat(inputs)
    seg_logit = self.decode_head.predict(x, batch_img_metas, self.test_cfg)

    ctx = FUNCTION_REWRITER.get_context()
    if get_codebase_config(ctx.cfg).get('with_argmax', True) is False:
        return seg_logit

    # mark seg_head
    @mark('decode_head', outputs=['output'])
    def __mark_seg_logit(seg_logit):
        return seg_logit

    seg_logit = __mark_seg_logit(seg_logit)

    seg_pred = seg_logit.argmax(dim=1, keepdim=True)
    return seg_pred
