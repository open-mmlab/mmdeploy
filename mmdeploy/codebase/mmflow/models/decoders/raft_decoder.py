# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import numpy as np
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmflow.models.decoders.raft_decoder.RAFTDecoder.forward_test')
def raft_decoder_forward_test(
        ctx,
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        flow: torch.Tensor,
        h_feat: torch.Tensor,
        cxt_feat: torch.Tensor,
        img_metas=None) -> Sequence[Dict[str, np.ndarray]]:
    """Forward function when model training.

    Args:
        feat1 (Tensor): The feature from the first input image.
        feat2 (Tensor): The feature from the second input image.
        flow (Tensor): The last estimated flow from GRU cell.
        h (Tensor): The hidden state for GRU cell.
        cxt_feat (Tensor): The contextual feature from the first image.
        img_metas (Sequence[dict], optional): meta data of image to revert
            the flow to original ground truth size. Defaults to None.

    Returns:
        Sequence[Dict[str, ndarray]]: The batch of predicted optical flow
            with the same size of images before augmentation.
    """
    flow_pred = self.forward(feat1, feat2, flow, h_feat, cxt_feat)

    flow_result = flow_pred[-1]
    # flow maps with the shape [H, W, 2]
    flow_result = flow_result.permute(0, 2, 3, 1)
    # unravel batch dim
    flow_result = [dict(flow=flow_result)]
    return self.get_flow(flow_result, img_metas=img_metas)
