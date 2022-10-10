# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet3d.core.post_processing import box3d_multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.centerpoint.CenterPoint.extract_pts_feat')
def centerpoint__extract_pts_feat(ctx, self, voxels, num_points, coors,
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

    voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
    batch_size = coors[-1, 0] + 1
    x = self.pts_middle_encoder(voxel_features, coors, batch_size)
    x = self.pts_backbone(x)
    if self.with_pts_neck:
        x = self.pts_neck(x)
    return x


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.centerpoint.CenterPoint.simple_test_pts')
def centerpoint__simple_test_pts(ctx, self, x, img_metas, rescale=False):
    """Rewrite this func to format model outputs.

    Args:
        x (torch.Tensor): Input points feature.
        img_metas (list[dict]): Meta information of samples.
        rescale (bool): Whether need rescale.

    Returns:
        List: Result of model.
    """
    outs = self.pts_bbox_head(x)
    bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
    return bbox_list


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.dense_heads.centerpoint_head.CenterHead.get_bboxes')
def centerpoint__get_bbox(ctx,
                          self,
                          preds_dicts,
                          img_metas,
                          img=None,
                          rescale=False):
    """Rewrite this func to format func inputs.

    Args
        pred_dicts (list[dict]): Each task predicts results.
        img_metas (list[dict]): Point cloud and image's meta info.
        img (torch.Tensor): Input image.
        rescale (Bool): Whether need rescale.

    Returns:
        list[dict]: Decoded bbox, scores and labels after nms.
    """
    rets = []
    for task_id, preds_dict in enumerate(preds_dicts):
        batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

        batch_reg = preds_dict[0]['reg']
        batch_hei = preds_dict[0]['height']

        if self.norm_bbox:
            batch_dim = torch.exp(preds_dict[0]['dim'])
        else:
            batch_dim = preds_dict[0]['dim']

        batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
        batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

        if 'vel' in preds_dict[0]:
            batch_vel = preds_dict[0]['vel']
        else:
            batch_vel = None
        temp = self.bbox_coder.decode(
            batch_heatmap,
            batch_rots,
            batch_rotc,
            batch_hei,
            batch_dim,
            batch_vel,
            reg=batch_reg,
            task_id=task_id)
        assert self.test_cfg['nms_type'] in ['circle', 'rotate']
        batch_bboxes = temp[0]['bboxes'].unsqueeze(0)
        batch_scores = temp[0]['scores'].unsqueeze(0).unsqueeze(-1)
        batch_cls_labels = temp[0]['labels'].unsqueeze(0).long()
        batch_bboxes_for_nms = batch_bboxes[..., [0, 1, 3, 4, 6]].clone()
        if self.test_cfg['nms_type'] == 'circle':
            raise NotImplementedError(
                'Not implement circle nms for deployment now!')
        else:
            rets.append(
                box3d_multiclass_nms(batch_bboxes, batch_bboxes_for_nms,
                                     batch_scores,
                                     self.test_cfg['score_threshold'],
                                     self.test_cfg['nms_thr'],
                                     self.test_cfg['post_max_size'], None,
                                     batch_cls_labels))

    # Merge branches results
    bboxes = torch.cat([ret[0] for ret in rets], dim=1)
    bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 5] * 0.5
    scores = torch.cat([ret[1] for ret in rets], dim=1)

    labels = [ret[3] for ret in rets]
    flag = 0
    for i, num_class in enumerate(self.num_classes):
        labels[i] += flag
        flag += num_class
    labels = torch.cat(labels, dim=1)
    return bboxes, scores, labels
