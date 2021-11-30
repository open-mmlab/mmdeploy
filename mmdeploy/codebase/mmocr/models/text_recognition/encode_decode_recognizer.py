# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.EncodeDecodeRecognizer.simple_test')
def encode_decode_recognizer__simple_test(ctx, self, img, img_metas, **kwargs):
    """Rewrite `simple_test` of EncodeDecodeRecognizer for default backend.

    Rewrite this function to early return the results to avoid post processing.
    The process is not suitable for exporting to backends and better get
    implemented in SDK.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class
            EncodeDecodeRecognizer.
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        img_metas (list[dict]): A list of image info dict where each dict
            has: 'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys, see
            :class:`mmdet.datasets.pipelines.Collect`.

    Returns:
        out_dec (Tensor): A feature map output from a decoder. The tensor shape
            (N, H, W).
    """
    feat = self.extract_feat(img)

    out_enc = None
    if self.encoder is not None:
        out_enc = self.encoder(feat, img_metas)

    out_dec = self.decoder(feat, out_enc, None, img_metas, train_mode=False)
    return out_dec
