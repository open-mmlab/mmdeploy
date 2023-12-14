# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from mmpose.structures.bbox import bbox_xyxy2cs
from torch import Tensor

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops.nms import multiclass_nms
from mmdeploy.utils import Backend, get_backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmpose.models.heads.hybrid_heads.'
    'rtmo_head.RTMOHead.forward')
def predict(self,
            x: Tuple[Tensor],
            batch_data_samples: List = [],
            test_cfg: Optional[dict] = None):
    """Get predictions and transform to bbox and keypoints results.
    Args:
        x (Tuple[Tensor]): The input tensor from upstream network.
        batch_data_samples: Batch image meta info. Defaults to None.
        test_cfg: The runtime config for testing process.

    Returns:
        Tuple[Tensor]: Predict bbox and keypoint results.
        - dets (Tensor): Predict bboxes and scores, which is a 3D Tensor,
            has shape (batch_size, num_instances, 5), the last dimension 5
            arrange as (x1, y1, x2, y2, score).
        - pred_kpts (Tensor): Predict keypoints and scores, which is a 4D
            Tensor, has shape (batch_size, num_instances, num_keypoints, 5),
            the last dimension 3 arrange as (x, y, score).
    """

    # deploy context
    ctx = FUNCTION_REWRITER.get_context()
    backend = get_backend(ctx.cfg)
    deploy_cfg = ctx.cfg

    cfg = self.test_cfg if test_cfg is None else test_cfg

    # get predictions
    cls_scores, bbox_preds, _, kpt_vis, pose_vecs = self.head_module(x)[:5]
    assert len(cls_scores) == len(bbox_preds)
    num_imgs = cls_scores[0].shape[0]

    # flatten and concat predictions
    scores = self._flatten_predictions(cls_scores).sigmoid()
    flatten_bbox_preds = self._flatten_predictions(bbox_preds)
    flatten_pose_vecs = self._flatten_predictions(pose_vecs)
    flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()
    bboxes = self.decode_bbox(flatten_bbox_preds, self.flatten_priors,
                              self.flatten_stride)

    if backend == Backend.TENSORRT:
        # pad for batched_nms because its output index is filled with -1
        bboxes = torch.cat(
            [bboxes,
             bboxes.new_zeros((bboxes.shape[0], 1, bboxes.shape[2]))],
            dim=1)

        scores = torch.cat(
            [scores, scores.new_zeros((scores.shape[0], 1, 1))], dim=1)

    # nms parameters
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.get('nms_thr', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.get('pre_top_k', -1)
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    # do nms
    _, _, nms_indices = multiclass_nms(
        bboxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k,
        output_index=True)

    batch_inds = torch.arange(num_imgs, device=scores.device).view(-1, 1)

    # filter predictions
    dets = torch.cat([bboxes, scores], dim=2)
    dets = dets[batch_inds, nms_indices, ...]
    pose_vecs = flatten_pose_vecs[batch_inds, nms_indices, ...]
    kpt_vis = flatten_kpt_vis[batch_inds, nms_indices, ...]
    grids = self.flatten_priors[nms_indices, ...]

    # decode keypoints
    bbox_cs = torch.cat(bbox_xyxy2cs(dets[..., :4], self.bbox_padding), dim=-1)
    keypoints = self.dcc.forward_test(pose_vecs, bbox_cs, grids)
    pred_kpts = torch.cat([keypoints, kpt_vis.unsqueeze(-1)], dim=-1)

    return dets, pred_kpts
