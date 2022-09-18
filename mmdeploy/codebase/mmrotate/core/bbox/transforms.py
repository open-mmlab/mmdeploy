# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmrotate.core.bbox.transforms import norm_angle

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmrotate.core.bbox.transforms.poly2obb_le90',
    backend='tensorrt')
def poly2obb_le90__tensorrt(ctx, polys: torch.Tensor) -> torch.Tensor:
    """This is a rewrite for poly2obb to remove NonZero ops.

    Args:
        ctx : context of the rewriter.
        polys (torch.Tensor): input

    Returns:
        torch.Tensor: output
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = torch.where(edge1 > edge2, angles1, angles2)
    angles = norm_angle(angles, 'le90')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmrotate.core.bbox.transforms.poly2obb_le135')
def poly2obb_le135__default(ctx, polys):
    """This is a rewrite for poly2obb to remove NonZero ops.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = torch.where(edge1 > edge2, angles1, angles2)
    angles = norm_angle(angles, 'le135')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmrotate.core.bbox.transforms.obb2poly_le135')
def obb2poly_le135__default(ctx, rboxes):
    """Support batched input.

    Args:
        ctx : context of rewriter
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    B, N = rboxes.shape[:2]
    x_ctr, y_ctr, width, height, angle = rboxes[..., 0], rboxes[
        ..., 1], rboxes[..., 2], rboxes[..., 3], rboxes[..., 4]
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=-1).reshape(B, N, 2, 4)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(B, N, 2, 2)
    polys = M.matmul(rects).permute(0, 1, 3, 2)
    xy_ctr = torch.stack([x_ctr, y_ctr], dim=-1).unsqueeze(-2)
    polys += xy_ctr
    polys = polys.reshape(B, N, 8)
    return polys.contiguous()
