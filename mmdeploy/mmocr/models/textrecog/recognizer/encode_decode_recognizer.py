from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.EncodeDecodeRecognizer.simple_test')
def simple_test_of_encode_decode_recognizer(ctx, self, img, img_metas,
                                            **kwargs):
    """Rewrite `forward` for default backend."""
    feat = self.extract_feat(img)

    out_enc = None
    if self.encoder is not None:
        out_enc = self.encoder(feat, img_metas)

    out_dec = self.decoder(feat, out_enc, None, img_metas, train_mode=False)
    return out_dec
