# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import get_ir_config


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.extract_img_feat'  # noqa: E501
)
def mvxtwostagedetector__extract_img_feat(self, img: torch.Tensor) -> dict:
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
def mvxtwostagedetector__extract_feat(self, batch_inputs_dict: dict) -> tuple:
    """Rewrite this func to remove voxelize op.

    Args:
        batch_inputs_dict (dict): Input dict comprises `voxels`, `num_points`
            and `coors`
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
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.forward')
def mvxtwostagedetector__forward(self, voxels: torch.Tensor,
                                 num_points: torch.Tensor, coors: torch.Tensor,
                                 **kwargs):
    """Rewrite this func to remove voxelize op.

    Args:
        voxels (Tensor): input voxels
        num_points (Tensor): input num_points
        coors (Tensor): input coors

    Returns:
        tuple: A tuple of classification scores, bbox and direction
            classification prediction.

            - cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, the channels number
                is num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, the channels number
                is num_base_priors * C.
            - dir_cls_preds (list[Tensor|None]): Direction classification
                predictions for all scale levels, each is a 4D-tensor,
                the channels number is num_base_priors * 2.
    """
    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg
    batch_inputs_dict = {
        'voxels': {
            'voxels': voxels,
            'num_points': num_points,
            'coors': coors
        }
    }

    _, pts_feats = self.extract_feat(batch_inputs_dict=batch_inputs_dict)
    outs = self.pts_bbox_head(pts_feats)

    if type(outs[0][0]) is dict:
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
    else:
        preds = []
        expect_names = []
        for i in range(len(outs[0])):
            preds += [outs[0][i], outs[1][i], outs[2][i]]
            expect_names += [
                f'cls_score{i}', f'bbox_pred{i}', f'dir_cls_pred{i}'
            ]
        # check if output_names is set correctly.
        onnx_cfg = get_ir_config(deploy_cfg)
        output_names = onnx_cfg['output_names']
        if output_names != list(expect_names):
            raise RuntimeError(f'`output_names` should be {expect_names} '
                               f'but given {output_names}\n'
                               f'Deploy config:\n{deploy_cfg.pretty_text}')
        return tuple(preds)
