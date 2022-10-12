# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter('mmpose.models.heads.BaseHead.decode')
def base_head__decode(ctx, self, batch_outputs):
    """Rewrite `decode` of BaseHead for default backend.

    1. to support exporting codecs like SimCCLabel

    Args:
        batch_outputs (tuple[Tensor]): Input features.

    Returns:
        keypoints (torch.Tensor): Output keypoints in shape
        of (N, 17, 3).
    """
    if self.decoder is None:
        raise RuntimeError(
            f'The decoder has not been set in {self.__class__.__name__}. '
            'Please set the decoder configs in the init parameters to '
            'enable head methods `head.predict()` and `head.decode()`')

    if self.decoder.support_batch_decoding:
        batch_keypoints, batch_scores = self.decoder.batch_decode(
            batch_outputs)

    else:
        batch_keypoints, batch_scores = self.decoder.decode(batch_outputs)
    if batch_scores.ndim == 2:
        batch_scores = batch_scores.unsqueeze(2)
    preds = torch.cat([batch_keypoints, batch_scores], dim=2)
    return preds
