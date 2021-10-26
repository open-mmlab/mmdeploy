import mmocr.utils as utils
import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.encoders.SAREncoder.forward',
    backend='default')
def forward_of_sar_encoder(ctx, self, feat, img_metas=None):
    """Rewrite `forward` of SAREncoder."""
    if img_metas is not None:
        assert utils.is_type_list(img_metas, dict)
        assert len(img_metas) == feat.size(0)

    valid_ratios = None
    if img_metas is not None:
        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
        ] if self.mask else None

    h_feat = int(feat.size(2))
    feat_v = F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
    feat_v = feat_v.squeeze(2)  # bsz * C * W
    feat_v = feat_v.permute(0, 2, 1).contiguous()  # bsz * W * C

    holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * C

    if valid_ratios is not None:
        valid_hf = []
        T = holistic_feat.size(1)
        for i, valid_ratio in enumerate(valid_ratios):
            # use torch.ceil to replace original math.ceil and if else in mmocr
            valid_step = torch.ceil(T * valid_ratio).long() - 1
            valid_hf.append(holistic_feat[i, valid_step, :])
        valid_hf = torch.stack(valid_hf, dim=0)
    else:
        valid_hf = holistic_feat[:, -1, :]  # bsz * C

    holistic_feat = self.linear(valid_hf)  # bsz * C

    return holistic_feat
