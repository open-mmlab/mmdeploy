# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmocr.structures import TextRecogDataSample

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textrecog.EncoderDecoderRecognizer.forward')
def encoder_decoder_recognizer__forward(self, batch_inputs: torch.Tensor,
                                        data_samples: TextRecogDataSample,
                                        **kwargs) -> TextRecogDataSample:
    """Rewrite `forward` of EncoderDecoderRecognizer for default backend.

    Rewrite this function to early return the results to avoid post processing.
    The process is not suitable for exporting to backends and better get
    implemented in SDK.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the class
            EncoderDecoderRecognizer.
        batch_inputs (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        data_samples (TextRecogDataSample): Containing meta information
            and gold annotations for each of the images. Defaults to None.

    Returns:
        out_dec (Tensor): A feature map output from a decoder. The tensor shape
            (N, H, W).
    """
    feat = self.extract_feat(batch_inputs)
    out_enc = None
    if self.with_encoder:
        out_enc = self.encoder(feat, data_samples)
    return self.decoder.predict(feat, out_enc, data_samples)
