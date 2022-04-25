# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmrotate.core import norm_angle

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmrotate.core.bbox.coder.delta_xywha_rbbox_coder.delta2bbox',
    backend='default')
def delta2bbox(ctx,
               rois,
               deltas,
               means=(0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               add_ctr_clamp=False,
               ctr_clamp=32,
               angle_range='oc',
               norm_factor=None,
               edge_swap=False,
               proj_xy=False):
    """Rewrite `delta2bbox` for default backend.

    Support batch bbox decoder.

    Args:
        ctx (ContextCaller): The context with additional information.
        rois (Tensor): Boxes to be transformed. Has shape (N, 5).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 5) or (N, 5). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor.
            Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
        angle_range (str, optional): Angle representations. Defaults to 'oc'.
        norm_factor (None|float, optional): Regularization factor of angle.
        edge_swap (bool, optional): Whether swap the edge if w < h.
            Defaults to False.
        proj_xy (bool, optional): Whether project x and y according to angle.
            Defaults to False.

    Return:
        bboxes (Tensor): Boxes with shape (N, num_classes * 5) or (N, 5),
            where 5 represent cx, cy, w, h, angle.
    """
    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 5))
    denorm_deltas = reshaped_deltas * stds + means

    dx = denorm_deltas[..., 0]
    dy = denorm_deltas[..., 1]
    dw = denorm_deltas[..., 2]
    dh = denorm_deltas[..., 3]
    da = denorm_deltas[..., 4]
    if norm_factor:
        da *= norm_factor * np.pi
    # Compute center of each roi

    px = rois[..., None, 0]
    py = rois[..., None, 1]
    # Compute width/height of each roi
    pw = rois[..., None, 2]
    ph = rois[..., None, 3]
    # Compute rotated angle of each roi
    pa = rois[..., None, 4]
    dx_width = pw * dx
    dy_height = ph * dy
    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    if proj_xy:
        gx = dx * pw * torch.cos(pa) - dy * ph * torch.sin(pa) + px
        gy = dx * pw * torch.sin(pa) + dy * ph * torch.cos(pa) + py
    else:
        gx = px + dx_width
        gy = py + dy_height
    # Compute angle
    ga = norm_angle(pa + da, angle_range)
    if max_shape is not None:
        gx = gx.clamp(min=0, max=max_shape[1] - 1)
        gy = gy.clamp(min=0, max=max_shape[0] - 1)

    if edge_swap:
        w_regular = torch.where(gw > gh, gw, gh)
        h_regular = torch.where(gw > gh, gh, gw)
        theta_regular = torch.where(gw > gh, ga, ga + np.pi / 2)
        theta_regular = norm_angle(theta_regular, angle_range)
        return torch.stack([gx, gy, w_regular, h_regular, theta_regular],
                           dim=-1).view_as(deltas)
    else:
        return torch.stack([gx, gy, gw, gh, ga], dim=-1).view(deltas.size())
