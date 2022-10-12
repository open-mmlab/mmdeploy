# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector._forward')
def mvxtwostagedetector__simple_test(ctx,
                                     self,
                                     batch_inputs_dict: dict,
                                     batch_input_metas,
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
        batch_inputs_dict=batch_inputs_dict,
        batch_input_metas=batch_input_metas)
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
