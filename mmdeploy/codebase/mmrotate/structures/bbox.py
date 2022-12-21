# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


def _dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).

    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.structures.bbox.box_converters.qbox2rbox')
def qbox2rbox__default(boxes: Tensor) -> Tensor:
    """Convert quadrilateral boxes to rotated boxes.

    Implement with PyTorch.
    """
    polys = boxes
    points = torch.reshape(polys, [*polys.shape[:-1], 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[..., 0], axis=-1), axis=-1) * 0.25
    cys = torch.unsqueeze(torch.sum(points[..., 1], axis=-1), axis=-1) * 0.25
    _ws = torch.unsqueeze(
        _dist_torch(points[..., 0, :], points[..., 1, :]), axis=-1)
    _hs = torch.unsqueeze(
        _dist_torch(points[..., 1, :], points[..., 2, :]), axis=-1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[..., 1, 0] - points[..., 0, 0]),
                    points[..., 1, 1] - points[..., 0, 1]),
        axis=-1)
    odd = torch.eq(torch.remainder((_thetas / (np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=-1)
    return rbboxes
