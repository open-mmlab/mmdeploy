# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.extract_img_feat'  # noqa: E501
)
def mvxtwostagedetector__extract_img_feat(ctx, self,
                                          img: torch.Tensor) -> dict:
    """Extract features of images."""
    if self.with_img_backbone and img is not None:
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        img_feats = self.img_backbone(img)
    else:
        return None
    if self.with_img_neck:
        img_feats = self.img_neck(img_feats)
    return img_feats


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.extract_feat')
def mvxtwostagedetector__extract_feat(ctx, self,
                                      batch_inputs_dict: dict) -> tuple:
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
    voxel_dict = batch_inputs_dict.get('voxels', None)
    imgs = batch_inputs_dict.get('imgs', None)
    points = batch_inputs_dict.get('points', None)
    img_feats = self.extract_img_feat(imgs)
    pts_feats = self.extract_pts_feat(
        voxel_dict, points=points, img_feats=img_feats)
    return (img_feats, pts_feats)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector._forward')
def mvxtwostagedetector__forward(ctx, self, batch_inputs_dict: dict,
                                 data_samples, **kwargs):
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
    _, pts_feats = self.extract_feat(batch_inputs_dict=batch_inputs_dict)
    outs = self.pts_bbox_head(pts_feats)
    cls_score, bbox_pred, dir_cls_pred = outs[0][0], outs[1][0], outs[2][0]
    return cls_score, bbox_pred, dir_cls_pred
