# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.codebase.mmdet.core.ops import (ncnn_detection_output_forward,
                                              ncnn_prior_box_forward)
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.SSDHead.get_bboxes', backend='ncnn')
def ssd_head__get_bboxes__ncnn(ctx,
                               self,
                               cls_scores,
                               bbox_preds,
                               img_metas,
                               with_nms=True,
                               cfg=None,
                               **kwargs):
    """Rewrite `get_bboxes` of SSDHead for NCNN backend.

    This rewriter using ncnn PriorBox and DetectionOutput layer to
    support dynamic deployment, and has higher speed.

    Args:
        ctx (ContextCaller): The context with additional information.
        cls_scores (list[Tensor]): Box scores for each level in the
            feature pyramid, has shape
            (N, num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each
            level in the feature pyramid, has shape
            (N, num_anchors * 4, H, W).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used.
            Default: None.

    Returns:
        Tensor: outputs, shape is [N, num_det, 6].
    """
    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)
    aspect_ratio = [
        ratio[ratio > 1].detach().cpu().numpy()
        for ratio in self.anchor_generator.ratios
    ]
    strides = self.anchor_generator.strides
    min_sizes = self.anchor_generator.base_sizes
    if is_dynamic_flag:
        max_sizes = min_sizes[1:] + img_metas[0]['img_shape'][0:1].tolist()
        img_height = img_metas[0]['img_shape'][0].item()
        img_width = img_metas[0]['img_shape'][1].item()
    else:
        max_sizes = min_sizes[1:] + img_metas[0]['img_shape'][0:1]
        img_height = img_metas[0]['img_shape'][0]
        img_width = img_metas[0]['img_shape'][1]

    # if no reshape, concat will be error in ncnn.
    mlvl_anchors = [
        ncnn_prior_box_forward(cls_scores[i], aspect_ratio[i], img_height,
                               img_width, strides[i][0], strides[i][1],
                               max_sizes[i:i + 1],
                               min_sizes[i:i + 1]).reshape(1, 2, -1)
        for i in range(num_levels)
    ]

    mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

    cfg = self.test_cfg if cfg is None else cfg
    assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)
    batch_size = 1

    mlvl_valid_bboxes = []
    mlvl_scores = []
    for level_id, cls_score, bbox_pred in zip(
            range(num_levels), mlvl_cls_scores, mlvl_bbox_preds):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(batch_size, -1,
                                                 self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        mlvl_valid_bboxes.append(bbox_pred)
        mlvl_scores.append(cls_score)

    # NCNN DetectionOutput layer uses background class at 0 position, but
    # in mmdetection, background class is at self.num_classes position.
    # We should adapt for ncnn.
    batch_mlvl_valid_bboxes = torch.cat(mlvl_valid_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    if self.use_sigmoid_cls:
        batch_mlvl_scores = batch_mlvl_scores.sigmoid()
    else:
        batch_mlvl_scores = batch_mlvl_scores.softmax(-1)
    batch_mlvl_anchors = torch.cat(mlvl_anchors, dim=2)
    batch_mlvl_scores = torch.cat([
        batch_mlvl_scores[:, :, self.num_classes:],
        batch_mlvl_scores[:, :, 0:self.num_classes]
    ],
                                  dim=2)
    batch_mlvl_valid_bboxes = batch_mlvl_valid_bboxes.reshape(
        batch_size, 1, -1)
    batch_mlvl_scores = batch_mlvl_scores.reshape(batch_size, 1, -1)
    batch_mlvl_anchors = batch_mlvl_anchors.reshape(batch_size, 2, -1)

    post_params = get_post_processing_params(deploy_cfg)
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    output__ncnn = ncnn_detection_output_forward(
        batch_mlvl_valid_bboxes, batch_mlvl_scores, batch_mlvl_anchors,
        score_threshold, iou_threshold, pre_top_k, keep_top_k,
        self.num_classes + 1)

    return output__ncnn
