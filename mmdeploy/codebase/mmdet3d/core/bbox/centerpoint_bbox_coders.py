# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.core.bbox.coders.centerpoint_bbox_coders.CenterPointBBoxCoder.'
    'decode')
def centerpointbboxcoder__decode(ctx,
                                 self,
                                 heat,
                                 rot_sine,
                                 rot_cosine,
                                 hei,
                                 dim,
                                 vel,
                                 reg=None,
                                 task_id=-1):
    """Decode bboxes. Rewrite this func for default backend.

    Args:
        heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
        rot_sine (torch.Tensor): Sine of rotation with the shape of
            [B, 1, W, H].
        rot_cosine (torch.Tensor): Cosine of rotation with the shape of
            [B, 1, W, H].
        hei (torch.Tensor): Height of the boxes with the shape
            of [B, 1, W, H].
        dim (torch.Tensor): Dim of the boxes with the shape of
            [B, 1, W, H].
        vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
        reg (torch.Tensor, optional): Regression value of the boxes in
            2D with the shape of [B, 2, W, H]. Default: None.
        task_id (int, optional): Index of task. Default: -1.

    Returns:
        list[dict]: Decoded boxes.
    """
    batch, cat, _, _ = heat.size()

    scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

    if reg is not None:
        reg = self._transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.max_num, 2)
        xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, self.max_num, 1) + 0.5
        ys = ys.view(batch, self.max_num, 1) + 0.5

    # rotation value and direction label
    rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
    rot_sine = rot_sine.view(batch, self.max_num, 1)

    rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
    rot_cosine = rot_cosine.view(batch, self.max_num, 1)
    rot = torch.atan2(rot_sine, rot_cosine)

    # height in the bev
    hei = self._transpose_and_gather_feat(hei, inds)
    hei = hei.view(batch, self.max_num, 1)

    # dim of the box
    dim = self._transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, self.max_num, 3)

    # class label
    clses = clses.view(batch, self.max_num).float()
    scores = scores.view(batch, self.max_num)

    xs = xs.view(
        batch, self.max_num,
        1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
    ys = ys.view(
        batch, self.max_num,
        1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

    if vel is None:  # KITTI FORMAT
        final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
    else:  # exist velocity, nuscene format
        vel = self._transpose_and_gather_feat(vel, inds)
        vel = vel.view(batch, self.max_num, 2)
        final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

    final_scores = scores
    final_preds = clses
    self.post_center_range = torch.tensor(
        self.post_center_range, device=heat.device)
    range_mask = torch.prod(
        torch.cat((final_box_preds[..., :3] >= self.post_center_range[:3],
                   final_box_preds[..., :3] <= self.post_center_range[3:]),
                  dim=-1),
        dim=-1).bool()
    final_box_preds = torch.where(
        range_mask.unsqueeze(-1), final_box_preds,
        torch.zeros(1, device=heat.device))
    final_scores = torch.where(range_mask, final_scores,
                               torch.zeros(1, device=heat.device))
    final_preds = torch.where(range_mask, final_preds,
                              torch.zeros(1, device=heat.device))
    predictions_dict = {
        'bboxes': final_box_preds[0],
        'scores': final_scores[0],
        'labels': final_preds[0],
    }
    return [predictions_dict]
