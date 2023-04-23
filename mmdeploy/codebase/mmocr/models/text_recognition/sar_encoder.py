# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from mmocr.structures import TextRecogDataSample

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.encoders.SAREncoder.forward',
    backend='default')
def sar_encoder__forward(
        self,
        feat: torch.Tensor,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None):
    """Rewrite `forward` of SAREncoder for default backend.

    Rewrite this function to:
    1. convert tuple value of feat.size to int, making model exportable.
    2. use torch.ceil to replace original math.ceil and if else in mmocr.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class SAREncoder.
        feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
        data_samples (list[TextRecogDataSample], optional): Batch of
            TextRecogDataSample, containing valid_ratio information.
            Defaults to None.

    Returns:
        holistic_feat (Tensor): A feature map output from SAREncoder. The shape
            [N, M].
    """
    if data_samples is not None:
        assert len(data_samples) == feat.size(0)

    valid_ratios = None
    if data_samples is not None:
        valid_ratios = [
            data_sample.get('valid_ratio', 1.0) for data_sample in data_samples
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
            valid_step = torch.tensor(T * valid_ratio).ceil().long() - 1
            valid_hf.append(holistic_feat[i, valid_step, :])
        valid_hf = torch.stack(valid_hf, dim=0)
    else:
        valid_hf = holistic_feat[:, -1, :]  # bsz * C

    holistic_feat = self.linear(valid_hf)  # bsz * C

    return holistic_feat
