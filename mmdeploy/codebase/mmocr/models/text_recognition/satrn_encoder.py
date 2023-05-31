# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List

from mmocr.structures import TextRecogDataSample
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.SATRNEncoder.forward')
def satrn_encoder__forward(
        self,
        feat: Tensor,
        data_samples: List[TextRecogDataSample] = None) -> Tensor:
    """Forward propagation of encoder.

    Args:
        feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
        data_samples (list[TextRecogDataSample]): Batch of
            TextRecogDataSample, containing `valid_ratio` information.
            Defaults to None.

    Returns:
        Tensor: A tensor of shape :math:`(N, T, D_m)`.
    """
    valid_ratio = 1.0
    feat = self.position_enc(feat)
    n, c, h, w = feat.size()
    mask = feat.new_zeros((n, h, w))
    valid_width = min(w, math.ceil(w * valid_ratio))
    mask[:, :, :valid_width] = 1
    mask = mask.view(n, h * w)
    feat = feat.view(n, c, h * w)

    output = feat.permute(0, 2, 1).contiguous()
    for enc_layer in self.layer_stack:
        output = enc_layer(output, h, w, mask)
    output = self.layer_norm(output)

    return output
