# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.dense_heads.anchor_free_mono3d_head.AnchorFreeMono3DHead.'
    '_get_points_single')
def anchorfreemono3dhead___get_points_single(ctx,
                                             self,
                                             featmap_size,
                                             stride,
                                             dtype,
                                             device,
                                             flatten=False):
    """Get points of a single scale level.

    Rewrite this func for some backends.
    """
    h, w = featmap_size
    x_range = torch.arange(w, device=device)
    x_range = x_range.to(dtype)
    y_range = torch.arange(h, device=device)
    y_range = y_range.to(dtype)
    y, x = torch.meshgrid(y_range, x_range)
    if flatten:
        y = y.flatten()
        x = x.flatten()
    return y, x
