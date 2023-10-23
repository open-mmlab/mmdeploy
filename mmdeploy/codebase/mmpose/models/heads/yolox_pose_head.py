# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops.nms import multiclass_nms
from mmdeploy.utils import Backend, get_backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmpose.models.heads.hybrid_heads.'
    'yoloxpose_head.YOLOXPoseHead.forward')
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

    cls_scores, objectnesses, bbox_preds, kpt_offsets, \
        kpt_vis = self.head_module(x)[:5]

    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg
    dtype = cls_scores[0].dtype
    device = cls_scores[0].device

    assert len(cls_scores) == len(bbox_preds)
    cfg = self.test_cfg if test_cfg is None else test_cfg

    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    self.mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=dtype, device=device)

    flatten_priors = torch.cat(self.mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full((featmap_size.numel(), ), stride)
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = self._flatten_predictions(cls_scores).sigmoid()
    flatten_bbox_preds = self._flatten_predictions(bbox_preds)
    flatten_objectness = self._flatten_predictions(objectnesses).sigmoid()
    flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
    flatten_kpt_vis = self._flatten_predictions(kpt_vis).sigmoid()
    bboxes = self.decode_bbox(flatten_bbox_preds, flatten_priors,
                              flatten_stride)
    flatten_decoded_kpts = self.decode_kpt_reg(flatten_kpt_offsets,
                                               flatten_priors, flatten_stride)

    scores = flatten_cls_scores * flatten_objectness

    pred_kpts = torch.cat([flatten_decoded_kpts,
                           flatten_kpt_vis.unsqueeze(3)],
                          dim=3)

    backend = get_backend(deploy_cfg)
    if backend == Backend.TENSORRT:
        # pad for batched_nms because its output index is filled with -1
        bboxes = torch.cat(
            [bboxes,
             bboxes.new_zeros((bboxes.shape[0], 1, bboxes.shape[2]))],
            dim=1)
        scores = torch.cat(
            [scores, scores.new_zeros((scores.shape[0], 1, 1))], dim=1)
        pred_kpts = torch.cat([
            pred_kpts,
            pred_kpts.new_zeros((pred_kpts.shape[0], 1, pred_kpts.shape[2],
                                 pred_kpts.shape[3]))
        ],
                              dim=1)

    # nms
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
    dets = torch.cat([bboxes, scores], dim=2)
    dets = dets[batch_inds, nms_indices, ...]
    pred_kpts = pred_kpts[batch_inds, nms_indices, ...]

    return dets, pred_kpts
