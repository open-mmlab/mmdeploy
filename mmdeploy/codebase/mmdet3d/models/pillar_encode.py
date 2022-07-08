# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.voxel_encoders.pillar_encoder.PillarFeatureNet.forward')
def pillar_encoder__forward(ctx, self, features, num_points, coors):
    """Rewrite this func to optimize node. Modify the code at
    _with_voxel_center and use slice instead of the original operation.

    Args:
        features (torch.Tensor): Point features or raw points in shape
            (N, M, C).
        num_points (torch.Tensor): Number of points in each pillar.
        coors (torch.Tensor): Coordinates of each voxel.

    Returns:
        torch.Tensor: Features of pillars.
    """
    features_ls = [features]
    # Find distance of x, y, and z from cluster center
    if self._with_cluster_center:
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

    # Find distance of x, y, and z from pillar center
    device = features.device

    if self._with_voxel_center:
        if not self.legacy:
            f_center = features[..., :3] - (coors[..., 1:] * torch.tensor(
                [self.vz, self.vy, self.vx]).to(device) + torch.tensor([
                    self.z_offset, self.y_offset, self.x_offset
                ]).to(device)).unsqueeze(1).flip(2)
        else:
            f_center = features[..., :3] - (coors[..., 1:] * torch.tensor(
                [self.vz, self.vy, self.vx]).to(device) + torch.tensor([
                    self.z_offset, self.y_offset, self.x_offset
                ]).to(device)).unsqueeze(1).flip(2)
            features_ls[0] = torch.cat((f_center, features[..., 3:]), dim=-1)
        features_ls.append(f_center)

    if self._with_distance:
        points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
        features_ls.append(points_dist)

    # Combine together feature decorations
    features = torch.cat(features_ls, dim=-1)
    # The feature decorations were calculated without regard to whether
    # pillar was empty. Need to ensure that
    # empty pillars remain set to zeros.
    voxel_count = features.shape[1]
    mask = get_paddings_indicator(num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(features)
    features *= mask
    for pfn in self.pfn_layers:
        features = pfn(features, num_points)

    return features.squeeze(1)
