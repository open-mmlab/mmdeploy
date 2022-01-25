import torch
from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.voxel_encoders.pillar_encoder.PillarFeatureNet.forward')
def forward(ctx, self, features, num_points, coors):
    """Forward function.

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
    dtype = features.dtype
    if self._with_voxel_center:
        if not self.legacy:
            f_center = torch.zeros_like(features[:, :, :2])
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        else:
            f_center = features[:, :, :2]
            f_center[:, :, 0] = f_center[:, :, 0] - (
                coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                self.x_offset)
            f_center[:, :, 1] = f_center[:, :, 1] - (
                coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                self.y_offset)
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
    mask = torch.unsqueeze(mask, -1).type_as(features).int()
    features *= mask

    for pfn in self.pfn_layers:
        features = pfn(features, num_points)

    return features.squeeze(1)
