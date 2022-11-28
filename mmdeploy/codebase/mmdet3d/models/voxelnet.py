# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.voxelnet.VoxelNet.simple_test')
def voxelnet__simple_test(ctx,
                          self,
                          voxels,
                          num_points,
                          coors,
                          img_metas=None,
                          imgs=None,
                          rescale=False):
    """Test function without augmentaiton. Rewrite this func to remove model
    post process.

    Args:
        voxels (torch.Tensor): Point features or raw points in shape (N, M, C).
        num_points (torch.Tensor): Number of points in each pillar.
        coors (torch.Tensor): Coordinates of each voxel.
        input_metas (list[dict]): Contain pcd meta info.

    Returns:
        List: Result of model.
    """
    x = self.extract_feat(voxels, num_points, coors, img_metas)
    outs = self.bbox_head(x)
    outs = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
    return outs


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.voxelnet.VoxelNet.extract_feat')
def voxelnet__extract_feat(ctx,
                           self,
                           voxels,
                           num_points,
                           coors,
                           img_metas=None):
    """Extract features from points. Rewrite this func to remove voxelize op.

    Args:
        voxels (torch.Tensor): Point features or raw points in shape (N, M, C).
        num_points (torch.Tensor): Number of points in each pillar.
        coors (torch.Tensor): Coordinates of each voxel.
        input_metas (list[dict]): Contain pcd meta info.

    Returns:
        torch.Tensor: Features from points.
    """
    voxel_features = self.voxel_encoder(voxels, num_points, coors)
    batch_size = coors[-1, 0] + 1  # refactor
    assert batch_size == 1
    x = self.middle_encoder(voxel_features, coors, batch_size)
    x = self.backbone(x)
    if self.with_neck:
        x = self.neck(x)
    return x
