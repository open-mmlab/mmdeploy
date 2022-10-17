# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch

from mmdeploy.codebase.mmdet import (gather_topk, get_post_processing_params,
                                     multiclass_nms,
                                     pad_with_value_if_necessary)
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


def _bbox_pre_decode(points: torch.Tensor, bbox_pred: torch.Tensor,
                     stride: torch.Tensor):
    """compute real bboxes."""
    points = points[..., :2]
    bbox_pos_center = torch.cat([points, points], dim=-1)
    bboxes = bbox_pred * stride + bbox_pos_center
    return bboxes


def _bbox_post_decode(bboxes: torch.Tensor, max_shape: Sequence[int]):
    """clamp bbox."""
    x1 = bboxes[..., 0].clamp(min=0, max=max_shape[1])
    y1 = bboxes[..., 1].clamp(min=0, max=max_shape[0])
    x2 = bboxes[..., 2].clamp(min=0, max=max_shape[1])
    y2 = bboxes[..., 3].clamp(min=0, max=max_shape[0])
    decoded_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return decoded_bboxes


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.RepPointsHead.points2bbox')
def reppoints_head__points2bbox(ctx, self, pts, y_first=True):
    """Rewrite of `points2bbox` in `RepPointsHead`.

    Use `self.moment_transfer` in `points2bbox` will cause error:
    RuntimeError: Input, output and indices must be on the current device
    """
    update_moment = hasattr(self, 'moment_transfer')
    if update_moment:
        moment_transfer = self.moment_transfer
        delattr(self, 'moment_transfer')
        self.moment_transfer = torch.tensor(moment_transfer.data)
    ret = ctx.origin_func(self, pts, y_first=y_first)
    if update_moment:
        self.moment_transfer = moment_transfer
    return ret


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.RepPointsHead.get_bboxes')
def reppoints_head__get_bboxes(ctx,
                               self,
                               cls_scores,
                               bbox_preds,
                               score_factors=None,
                               img_metas=None,
                               cfg=None,
                               rescale=None,
                               **kwargs):
    """Rewrite `get_bboxes` of `RepPointsHead` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (RepPointsHead): The instance of the class RepPointsHead.
        cls_scores (list[Tensor]): Box scores for each scale level
            with shape (N, num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W).
        score_factors (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W). Default None.
        img_metas (list[dict]):  Meta information of the image, e.g.,
            image size, scaling factor, etc.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.
        rescale (bool): If True, return boxes in original image space.
            Default: False.

    Returns:
        tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
    """
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
    mlvl_priors = [priors.unsqueeze(0) for priors in mlvl_priors]

    mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    assert img_metas is not None
    img_shape = img_metas[0]['img_shape']

    assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
    batch_size = cls_scores[0].shape[0]
    cfg = self.test_cfg
    pre_topk = cfg.get('nms_pre', -1)

    mlvl_valid_bboxes = []
    mlvl_valid_scores = []

    for level_idx, (cls_score, bbox_pred, priors) in enumerate(
            zip(mlvl_cls_scores, mlvl_bbox_preds, mlvl_priors)):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                       self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(-1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        if not is_dynamic_flag:
            priors = priors.data
        if pre_topk > 0:
            priors = pad_with_value_if_necessary(priors, 1, pre_topk)
            bbox_pred = pad_with_value_if_necessary(bbox_pred, 1, pre_topk)
            scores = pad_with_value_if_necessary(scores, 1, pre_topk, 0.)

            nms_pre_score = scores

            # Get maximum scores for foreground classes.
            if self.use_sigmoid_cls:
                max_scores, _ = nms_pre_score.max(-1)
            else:
                max_scores, _ = nms_pre_score[..., :-1].max(-1)
            _, topk_inds = max_scores.topk(pre_topk)
            bbox_pred, scores = gather_topk(
                bbox_pred,
                scores,
                inds=topk_inds,
                batch_size=batch_size,
                is_batched=True)
            priors = gather_topk(
                priors,
                inds=topk_inds,
                batch_size=batch_size,
                is_batched=False)

        bbox_pred = _bbox_pre_decode(priors, bbox_pred,
                                     self.point_strides[level_idx])
        mlvl_valid_bboxes.append(bbox_pred)
        mlvl_valid_scores.append(scores)

    batch_mlvl_bboxes_pred = torch.cat(mlvl_valid_bboxes, dim=1)
    batch_scores = torch.cat(mlvl_valid_scores, dim=1)
    batch_bboxes = _bbox_post_decode(
        bboxes=batch_mlvl_bboxes_pred, max_shape=img_shape)

    if not self.use_sigmoid_cls:
        batch_scores = batch_scores[..., :self.num_classes]

    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    return multiclass_nms(
        batch_bboxes,
        batch_scores,
        max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)
