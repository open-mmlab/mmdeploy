from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.decoders.CRNNDecoder.forward_train',
    backend='ncnn')
def forward_train_of_crnndecoder(ctx, self, feat, out_enc, targets_dict,
                                 img_metas):
    assert feat.size(2) == 1, 'feature height must be 1'
    if self.rnn_flag:
        x = feat.squeeze(2)  # [N, C, W]
        x = x.permute(0, 2, 1)  # [N, W, C]
        outputs = self.decoder(x)
    else:
        x = self.decoder(feat)
        x = x.permute(0, 3, 1, 2).contiguous()
        n, w, c, h = x.size()
        outputs = x.view(n, w, c * h)
    return outputs
