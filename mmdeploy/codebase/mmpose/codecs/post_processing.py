# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmpose.codecs.utils.post_processing.get_simcc_maximum',
    backend='default')
def get_simcc_maximum(ctx, simcc_x: Tensor,
                      simcc_y: Tensor) -> Tuple[Tensor, Tensor]:
    """Get maximum response location and value from simcc representations.
    1. rewrite to support Tensor inputs
    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (Tensor): x-axis SimCC in shape (N, K, Wx)
        simcc_y (Tensor): y-axis SimCC in shape (N, K, Wy)

    Returns:
        tuple:
        - locs (Tensor): locations of maximum heatmap responses in shape
            (N, K, 2)
        - vals (Tensor): values of maximum heatmap responses in shape
            (N, K, 1)
    """

    assert isinstance(simcc_x, Tensor), 'simcc_x should be Tensor'
    assert isinstance(simcc_y, Tensor), 'simcc_y should be Tensor'
    assert simcc_x.ndim == simcc_y.ndim == 3
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = torch.argmax(simcc_x, dim=1, keepdim=True)
    y_locs = torch.argmax(simcc_y, dim=1, keepdim=True)

    locs = torch.cat([x_locs, y_locs], dim=1).to(simcc_x)

    max_val_x, _ = torch.max(simcc_x, dim=1)
    max_val_y, _ = torch.max(simcc_y, dim=1)

    vals = torch.where(max_val_x > max_val_y, max_val_y, max_val_x)
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K, 1)
    return locs, vals
