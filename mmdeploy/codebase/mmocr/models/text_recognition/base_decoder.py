# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch
from mmocr.structures import TextRecogDataSample

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.decoders.BaseDecoder.predict')
def base_decoder__forward(
    self,
    feat: Optional[torch.Tensor] = None,
    out_enc: Optional[torch.Tensor] = None,
    data_samples: Optional[Sequence[TextRecogDataSample]] = None
) -> Sequence[TextRecogDataSample]:
    """Rewrite `predict` of `BaseDecoder` to skip post-process.

    Args:
        feat (Tensor, optional): Features from the backbone. Defaults
            to None.
        out_enc (Tensor, optional): Features from the encoder. Defaults
            to None.
        data_samples (list[TextRecogDataSample]): A list of N datasamples,
            containing meta information and gold annotations for each of
            the images. Defaults to None.
    """
    out_dec = self(feat, out_enc, data_samples)
    return out_dec
