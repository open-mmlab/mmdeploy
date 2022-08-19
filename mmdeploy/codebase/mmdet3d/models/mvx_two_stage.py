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


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.'
    'extract_pts_feat')
def mvxtwostagedetector__extract_pts_feat(ctx, self, voxels, num_points, coors,
                                          img_feats, img_metas):
    """Extract features from points. Rewrite this func to remove voxelize op.

    Args:
        voxels (torch.Tensor): Point features or raw points in shape (N, M, C).
        num_points (torch.Tensor): Number of points in each voxel.
        coors (torch.Tensor): Coordinates of each voxel.
        img_feats (list[torch.Tensor], optional): Image features used for
            multi-modality fusion. Defaults to None.
        img_metas (list[dict]): Meta information of samples.

    Returns:
        torch.Tensor: Points feature.
    """
    if not self.with_pts_bbox:
        return None
    voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                            img_feats, img_metas)
    batch_size = coors[-1, 0] + 1
    x = self.pts_middle_encoder(voxel_features, coors, batch_size)
    x = self.pts_backbone(x)
    if self.with_pts_neck:
        x = self.pts_neck(x)
    return x


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.'
    'simple_test_pts')
def mvxtwostagedetector__simple_test_pts(ctx,
                                         self,
                                         x,
                                         img_metas,
                                         rescale=False):
    """Rewrite this func to format model outputs.

    Args:
        x (torch.Tensor): Input points feature.
        img_metas (list[dict]): Meta information of samples.
        rescale (bool): Whether need rescale.

    Returns:
        List: Result of model.
    """
    bbox_preds, scores, dir_scores = self.pts_bbox_head(x)
    return bbox_preds, scores, dir_scores
