# Copyright (c) OpenMMLab. All rights reserved.
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
    x = self.extract_feat(batch_inputs)
    seg_logit = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
    seg_pred = seg_logit.argmax(dim=1, keepdim=True)
    return seg_pred
