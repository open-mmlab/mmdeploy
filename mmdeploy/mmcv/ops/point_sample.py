# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.ops.point_sample', backend='default')
def point_sample__default(ctx, input, points, align_corners=False, **kwargs):
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (torch.Tensor): Feature map, shape (N, C, H, W).
        points (torch.Tensor): Image based absolute point coordinates
            (normalized), range [0, 1] x [0, 1], shape (N, P, 2) or
            (N, Hgrid, Wgrid, 2).
        align_corners (bool, optional): Whether align_corners.
            Default: False

    Returns:
        torch.Tensor: Features of `point` on `input`, shape (N, C, P) or
        (N, C, Hgrid, Wgrid).
    """
    from mmcv.ops.point_sample import denormalize
    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.ops.SimpleRoIAlign.forward')
def simple_roialign__forward(ctx, self, features, rois):
    """Rewrite `forward` of SimpleRoIAlign.

    Args:
        features (torch.Tensor): Feature map, shape (N, C, H, W).
        rois (torch.Tensor):

    Returns:
        torch.Tensor: RoI features.
    """
    from mmcv.ops.point_sample import (generate_grid, point_sample,
                                       rel_roi_point_to_rel_img_point)
    num_imgs = features.size(0)
    num_rois = rois.size(0)
    rel_roi_points = generate_grid(
        num_rois, self.output_size, device=rois.device)
    rel_img_points = rel_roi_point_to_rel_img_point(rois, rel_roi_points,
                                                    features,
                                                    self.spatial_scale)
    rel_img_points = rel_img_points.reshape(num_imgs, -1,
                                            *rel_img_points.shape[1:])
    point_feats = point_sample(
        features, rel_img_points, align_corners=not self.aligned)
    point_feats = point_feats.transpose(1, 2)

    channels = features.size(1)
    roi_feats = point_feats.reshape(num_rois, channels, *self.output_size)

    return roi_feats
