# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.decode_heads.ema_head.EMAModule.forward')
def ema_module__forward(ctx, self, feats):
    """Rewrite `forward` for default backend.

    Replace torch.einsum with other operations.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        feats (Tensor): Input feature.

    Returns:
        torch.Tensor: Output feature.
    """
    batch_size, channels, height, width = feats.size()
    # [batch_size, channels, height*width]
    feats = feats.view(batch_size, channels, height * width)
    # [batch_size, channels, num_bases]
    bases = self.bases.repeat(batch_size, 1, 1)

    with torch.no_grad():
        for i in range(self.num_stages):
            # [batch_size, height*width, num_bases]
            attention = torch.bmm(feats.transpose(1, 2), bases)
            # attention = torch.einsum('bcn,bck->bnk', feats, bases)
            attention = F.softmax(attention, dim=2)
            # l1 norm
            attention_normed = F.normalize(attention, dim=1, p=1)
            # [batch_size, channels, num_bases]
            bases = torch.bmm(feats, attention_normed)
            # bases = torch.einsum('bcn,bnk->bck', feats, attention_normed)
            # l2 norm
            bases = F.normalize(bases, dim=1, p=2)
    feats_recon = torch.bmm(bases, attention.transpose(1, 2))
    # feats_recon = torch.einsum('bck,bnk->bcn', bases, attention)
    feats_recon = feats_recon.view(batch_size, channels, height, width)
    return feats_recon
