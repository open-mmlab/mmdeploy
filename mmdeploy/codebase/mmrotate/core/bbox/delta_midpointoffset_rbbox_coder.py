# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmrotate.core import norm_angle

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmrotate.core.bbox.coder.delta_midpointoffset_rbbox_coder'
    '.delta2bbox',
    backend='default')
def delta2bbox(ctx,
               rois,
               deltas,
               means=(0., 0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1., 1.),
               wh_ratio_clip=16 / 1000,
               version='oc'):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas
    are network outputs used to shift/scale those boxes. This is the inverse
    function of :func:`bbox2delta`.


    Args:
        rois (torch.Tensor): Boxes to be transformed. Has shape (N, 6).
        deltas (torch.Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 6) or (N, 6). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1., 1.).
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        version (str, optional): Angle representations. Defaults to 'oc'.

    Returns:
        Tensor: Boxes with shape (N, num_classes * 5) or (N, 5), where 5
           represent cx, cy, w, h, a.
    """
    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 6))
    denorm_deltas = reshaped_deltas * stds + means

    # means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 6)
    # stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 6)
    # denorm_deltas = deltas * stds + means
    dx = denorm_deltas[..., 0::6]
    dy = denorm_deltas[..., 1::6]
    dw = denorm_deltas[..., 2::6]
    dh = denorm_deltas[..., 3::6]
    da = denorm_deltas[..., 4::6]
    db = denorm_deltas[..., 5::6]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[..., None, None, 0] + rois[..., None, None, 2]) * 0.5)
    py = ((rois[..., None, None, 1] + rois[..., None, None, 3]) * 0.5)
    # Compute width/height of each roi
    pw = (rois[..., None, None, 2] - rois[..., None, None, 0])
    ph = (rois[..., None, None, 3] - rois[..., None, None, 1])
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    da = da.clamp(min=-0.5, max=0.5)
    db = db.clamp(min=-0.5, max=0.5)
    ga = gx + da * gw
    _ga = gx - da * gw
    gb = gy + db * gh
    _gb = gy - db * gh
    polys = torch.stack([ga, y1, x2, gb, _ga, y2, x1, _gb], dim=-1)

    center = torch.stack([gx, gy, gx, gy, gx, gy, gx, gy], dim=-1)
    center_polys = polys - center
    diag_len = torch.sqrt(center_polys[..., 0::2] * center_polys[..., 0::2] +
                          center_polys[..., 1::2] * center_polys[..., 1::2])
    max_diag_len, _ = torch.max(diag_len, dim=-1, keepdim=True)
    diag_scale_factor = max_diag_len / diag_len
    center_polys = center_polys * diag_scale_factor.transpose(3, 4).repeat(
        1, 1, 1, 1, 2).view(1, 5, 1, -1, 1).transpose(3, 4)
    rectpolys = center_polys + center

    # poly2obb
    polys = torch.reshape(rectpolys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan(
        (pt2[..., 1] - pt1[..., 1]) / (pt2[..., 0] - pt1[..., 0] + 1e-6))
    angles2 = torch.atan(
        (pt4[..., 1] - pt1[..., 1]) / (pt4[..., 0] - pt1[..., 0] + 1e-6))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, version)
    angles[angles > 1.57] = -angles[angles > 1.57]
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    obboxes = torch.stack([x_ctr, y_ctr, width, height, angles], 1)

    return obboxes.view(delta_shape[:-1] + (5, ))
