# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence

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
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.extract_pts_feat'  # noqa: E501
)
def mvxtwostagedetector__extract_pts_feat(
        ctx,
        self,
        voxel_dict: Dict[str, torch.Tensor],
        points: Optional[List[torch.Tensor]] = None,
        img_feats: Optional[Sequence[torch.Tensor]] = None,
        batch_input_metas: Optional[List[dict]] = None
) -> Sequence[torch.Tensor]:
    """Extract features of points.

    Args:
        voxel_dict(Dict[str, torch.Tensor]): Dict of voxelization infos.
        points (List[torch.Tensor], optional):  Point cloud of multiple inputs.
        img_feats (list[torch.Tensor], tuple[tensor], optional): Features from
            image backbone.
        batch_input_metas (list[dict], optional): The meta information
            of multiple samples. Defaults to True.

    Returns:
        Sequence[tensor]: points features of multiple inputs
        from backbone or neck.
    """

    if not self.with_pts_bbox:
        return None
    voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'], img_feats,
                                            batch_input_metas)
    batch_size = voxel_dict['coors'][-1, 0] + 1
    x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
    x = self.pts_backbone(x)
    if self.with_pts_neck:
        x = self.pts_neck(x)
    return x


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
def mvxtwostagedetector__forward(ctx, self, batch_inputs_dict: dict, **kwargs):
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
    bbox_preds, scores, dir_scores = [], [], []
    for task_res in outs:
        bbox_preds.append(task_res[0]['reg'])
        bbox_preds.append(task_res[0]['height'])
        bbox_preds.append(task_res[0]['dim'])
        if 'vel' in task_res[0].keys():
            bbox_preds.append(task_res[0]['vel'])
        scores.append(task_res[0]['heatmap'])
        dir_scores.append(task_res[0]['rot'])
    bbox_preds = torch.cat(bbox_preds, dim=1)
    scores = torch.cat(scores, dim=1)
    dir_scores = torch.cat(dir_scores, dim=1)
    return scores, bbox_preds, dir_scores
