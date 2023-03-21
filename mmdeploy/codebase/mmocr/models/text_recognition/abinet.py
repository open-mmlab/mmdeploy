# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.ABINet.simple_test')
def abinet__simple_test(ctx, self, img, img_metas, **kwargs):
    """Rewrite `simple_test` of ABINet for default backend.

    Rewrite this function to early return the results to avoid post processing.
    The process is not suitable for exporting to backends and better get
    implemented in SDK.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class
            ABINet.
        img (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        img_metas (list[dict]): A list of image info dict where each dict
            has: 'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys, see
            :class:`mmdet.datasets.pipelines.Collect`.

    Returns:
        logits (Tensor): The logits of the ABIFuser. The tensor shape
            (N, max_seq_len, num_chars).
    """
    feat = self.extract_feat(img)

    text_logits = None
    out_enc = None
    if self.encoder is not None:
        out_enc = self.encoder(feat)
        text_logits = out_enc['logits']

    out_decs = []
    out_fusers = []
    for _ in range(self.iter_size):
        if self.decoder is not None:
            out_dec = self.decoder(
                feat, text_logits, img_metas=img_metas, train_mode=False)
            out_decs.append(out_dec)

        if self.fuser is not None:
            out_fuser = self.fuser(out_enc['feature'], out_dec['feature'])
            text_logits = out_fuser['logits']
            out_fusers.append(out_fuser)

    if len(out_fusers) > 0:
        ret = out_fusers[-1]
    elif len(out_decs) > 0:
        ret = out_decs[-1]
    else:
        ret = out_enc
    return ret['logits']
