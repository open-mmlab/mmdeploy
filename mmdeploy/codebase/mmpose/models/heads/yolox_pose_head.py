# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from mmengine.config import ConfigDict
from torch import Tensor

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops.nms import multiclass_nms


@FUNCTION_REWRITER.register_rewriter(func_name='models.yolox_pose_head.'
                                     'YOLOXPoseHead.predict')
def predict(self,
            x: Tuple[Tensor],
            batch_data_samples=None,
            rescale: bool = True):
    outs = self(x)
    predictions = self.predict_by_feat(
        *outs, batch_img_metas=batch_data_samples, rescale=rescale)
    return predictions


@FUNCTION_REWRITER.register_rewriter(func_name='models.yolox_pose_head.'
                                     'YOLOXPoseHead.predict_by_feat')
def yolox_pose_head__predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        objectnesses: Optional[List[Tensor]] = None,
        kpt_preds: Optional[List[Tensor]] = None,
        vis_preds: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = True,
        with_nms: bool = True) -> Tuple[Tensor]:
    """Transform a batch of output features extracted by the head into bbox and
    keypoint results.

    In addition to the base class method, keypoint predictions are also
    calculated in this method.
    """
    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg
    dtype = cls_scores[0].dtype
    device = cls_scores[0].device
    bbox_decoder = self.bbox_coder.decode

    assert len(cls_scores) == len(bbox_preds)
    cfg = self.test_cfg if cfg is None else cfg

    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    self.mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=dtype, device=device)

    flatten_priors = torch.cat(self.mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size[0] * featmap_size[1] * self.num_base_priors, ),
            stride)
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
        for cls_score in cls_scores
    ]
    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    if objectnesses is not None:
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

    scores = cls_scores
    bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds,
                          flatten_stride)

    # deal with key-poinsts
    priors = torch.cat(self.mlvl_priors)
    strides = [
        priors.new_full((featmap_size.numel() * self.num_base_priors, ),
                        stride)
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    strides = torch.cat(strides)
    kpt_preds = torch.cat([
        kpt_pred.permute(0, 2, 3, 1).reshape(
            num_imgs, -1, self.num_keypoints * 2) for kpt_pred in kpt_preds
    ],
                          dim=1)
    flatten_decoded_kpts = self.decode_pose(priors, kpt_preds, strides)

    vis_preds = torch.cat([
        vis_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_keypoints,
                                             1) for vis_pred in vis_preds
    ],
                          dim=1).sigmoid()

    pred_kpts = torch.cat([flatten_decoded_kpts, vis_preds], dim=3)

    # nms
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.get('pre_top_k', -1)
    keep_top_k = post_params.get('keep_top_k', -1)
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
    bboxes = bboxes[batch_inds, nms_indices, ...]
    scores = scores[batch_inds, nms_indices, ...]
    dets = torch.cat([bboxes, scores], dim=2)
    pred_kpts = pred_kpts[batch_inds, nms_indices, ...]

    return dets, pred_kpts
