# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet3d.core import circle_nms

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
    _, pts_feats = self.extract_feat(
        voxels, num_points, coors, img=img, img_metas=img_metas)
    if pts_feats and self.with_pts_bbox:
        bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
    return bbox_pts


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector.extract_feat')
def mvxtwostagedetector__extract_feat(ctx, self, voxels, num_points, coors,
                                      img, img_metas):
    img_feats = self.extract_img_feat(img, img_metas)
    pts_feats = self.extract_pts_feat(voxels, num_points, coors, img_feats,
                                      img_metas)
    return (img_feats, pts_feats)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.centerpoint.CenterPoint.extract_pts_feat')
def centerpoint__extract_pts_feat(ctx, self, voxels, num_points, coors,
                                  img_feats, img_metas):
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
    outs = self.pts_bbox_head(x)
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


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.dense_heads.centerpoint_head.CenterHead.get_bboxes')
def centerpoint__get_bbox(ctx,
                          self,
                          cls_scores,
                          bbox_preds,
                          dir_scores,
                          img_metas,
                          img=None,
                          rescale=False):
    rets = []
    batch_size = 1
    scores_range = [0]
    bbox_range = []
    dir_range = []
    for i in range(len(self.num_classes)):
        scores_range.append(scores_range[i] + self.num_classes[i])
        bbox_range.append(i * 8)
        dir_range.append(i * 2)
    for task_id in range(len(self.num_classes)):
        num_class_with_bg = self.num_classes[task_id]

        batch_heatmap = cls_scores[
            0][:, scores_range[task_id]:scores_range[task_id + 1],
               ...].sigmoid()

        batch_reg = bbox_preds[0][:,
                                  bbox_range[task_id]:bbox_range[task_id] + 2,
                                  ...]
        batch_hei = bbox_preds[0][:, bbox_range[task_id] +
                                  2:bbox_range[task_id] + 3, ...]

        if self.norm_bbox:
            batch_dim = torch.exp(bbox_preds[0][:, bbox_range[task_id] +
                                                3:bbox_range[task_id] + 6,
                                                ...])
        else:
            batch_dim = bbox_preds[0][:, bbox_range[task_id] +
                                      3:bbox_range[task_id] + 6, ...]

        batch_vel = bbox_preds[0][:, bbox_range[task_id] +
                                  6:bbox_range[task_id] + 8, ...]

        batch_rots = dir_scores[0][:,
                                   dir_range[task_id]:dir_range[task_id] + 2,
                                   ...][:, 0].unsqueeze(1)
        batch_rotc = dir_scores[0][:,
                                   dir_range[task_id]:dir_range[task_id] + 2,
                                   ...][:, 1].unsqueeze(1)

        # if 'vel' in preds_dict[0]:
        #     batch_vel = preds_dict[0]['vel']
        # else:
        #     batch_vel = None

        temp = self.bbox_coder.decode(
            batch_heatmap,
            batch_rots,
            batch_rotc,
            batch_hei,
            batch_dim,
            batch_vel,
            reg=batch_reg,
            task_id=task_id)
        assert self.test_cfg['pts']['nms_type'] in ['circle', 'rotate']
        batch_reg_preds = [box['bboxes'] for box in temp]
        batch_cls_preds = [box['scores'] for box in temp]
        batch_cls_labels = [box['labels'] for box in temp]
        if self.test_cfg['pts']['nms_type'] == 'circle':
            ret_task = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                centers = boxes3d[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                keep = torch.tensor(
                    circle_nms(
                        boxes.detach().cpu().numpy(),
                        self.test_cfg['pts']['min_radius'][task_id],
                        post_max_size=self.test_cfg['pts']['post_max_size']),
                    dtype=torch.long,
                    device=boxes.device)

                boxes3d = boxes3d[keep]
                scores = scores[keep]
                labels = labels[keep]
                ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_task.append(ret)
            rets.append(ret_task)
        else:
            rets.append(
                self.get_task_detections(num_class_with_bg, batch_cls_preds,
                                         batch_reg_preds, batch_cls_labels,
                                         img_metas))

    # Merge branches results
    num_samples = len(rets[0])

    ret_list = []
    for i in range(num_samples):
        for k in rets[0][i].keys():
            if k == 'bboxes':
                bboxes = torch.cat([ret[i][k] for ret in rets])
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                bboxes = img_metas[i]['box_type_3d'](bboxes,
                                                     self.bbox_coder.code_size)
            elif k == 'scores':
                scores = torch.cat([ret[i][k] for ret in rets])
            elif k == 'labels':
                flag = 0
                for j, num_class in enumerate(self.num_classes):
                    rets[j][i][k] += flag
                    flag += num_class
                labels = torch.cat([ret[i][k].int() for ret in rets])
        ret_list.append([bboxes, scores, labels])
    return ret_list
