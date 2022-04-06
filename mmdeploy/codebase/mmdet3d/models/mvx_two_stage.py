# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.simple_test')
def mvxtwostagedetector__simple_test(ctx,
                                     self,
                                     voxels,
                                     num_points,
                                     coors,
                                     img_metas,
                                     img=None,
                                     rescale=False):
    """Rewrite this func to remove voxelize op.

    Args:
        voxels (torch.Tensor): Point features or raw points in shape (N, M, C).
        num_points (torch.Tensor): Number of points in each voxel.
        coors (torch.Tensor): Coordinates of each voxel.
        img_metas (list[dict]): Meta information of samples.
        img (torch.Tensor): Input image.
        rescale (Bool): Whether need rescale.

    Returns:
        list[dict]: Decoded bbox, scores and labels after nms.
    """
    _, pts_feats = self.extract_feat(
        voxels, num_points, coors, img=img, img_metas=img_metas)
    if pts_feats and self.with_pts_bbox:
        bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
    return bbox_pts


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.extract_feat')
def mvxtwostagedetector__extract_feat(ctx, self, voxels, num_points, coors,
                                      img, img_metas):
    """Rewrite this func to remove voxelize op.

    Args:
        voxels (torch.Tensor): Point features or raw points in shape (N, M, C).
        num_points (torch.Tensor): Number of points in each voxel.
        coors (torch.Tensor): Coordinates of each voxel.
        img (torch.Tensor): Input image.
        img_metas (list[dict]): Meta information of samples.

    Returns:
        tuple(torch.Tensor) : image feature and points feather.
    """
    img_feats = self.extract_img_feat(img, img_metas)
    pts_feats = self.extract_pts_feat(voxels, num_points, coors, img_feats,
                                      img_metas)
    return (img_feats, pts_feats)
