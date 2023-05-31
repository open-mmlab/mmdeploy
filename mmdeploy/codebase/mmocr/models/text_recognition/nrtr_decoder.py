# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.NRTRDecoder._get_source_mask')
def nrtr_decoder___get_source_mask(
        self, src_seq: torch.Tensor,
        valid_ratios: Sequence[float]) -> torch.Tensor:
    """Generate mask for source sequence.

    Args:
        src_seq (torch.Tensor): Image sequence. Shape :math:`(N, T, C)`.
        valid_ratios (list[float]): The valid ratio of input image. For
            example, if the width of the original image is w1 and the width
            after padding is w2, then valid_ratio = w1/w2. Source mask is
            used to cover the area of the padding region.

    Returns:
        Tensor or None: Source mask. Shape :math:`(N, T)`. The region of
        padding area are False, and the rest are True.
    """

    N, T, _ = src_seq.size()
    mask = None
    if len(valid_ratios) > 0:
        mask = src_seq.new_zeros((N, T), device=src_seq.device)
        valid_width = min(T, math.ceil(T * valid_ratios[0]))
        mask[:, :valid_width] = 1

    return mask
