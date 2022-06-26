# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet import (get_post_processing_params,
                                     pad_with_value_if_necessary)
from mmdeploy.codebase.mmrotate.core.post_processing import \
    fake_multiclass_nms_rotated
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.dense_heads.OrientedRPNHead.get_bboxes')
def oriented_rpn_head__get_bboxes(ctx,
                                  self,
                                  cls_scores,
                                  bbox_preds,
                                  score_factors=None,
                                  img_metas=None,
                                  cfg=None,
                                  rescale=False,
                                  with_nms=True,
                                  **kwargs):
    """Rewrite `get_bboxes` of `RPNHead` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (FoveaHead): The instance of the class FoveaHead.
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
            Default False.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.
    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor, Tensor]: batch_mlvl_bboxes, batch_mlvl_scores
    """
    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)

    device = cls_scores[0].device
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device=device)

    mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)

    cfg = self.test_cfg if cfg is None else cfg
    batch_size = mlvl_cls_scores[0].shape[0]
    pre_topk = cfg.get('nms_pre', -1)

    # loop over features, decode boxes
    mlvl_valid_bboxes = []
    mlvl_scores = []
    mlvl_valid_anchors = []
    for level_id, cls_score, bbox_pred, anchors in zip(
            range(num_levels), mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        cls_score = cls_score.permute(0, 2, 3, 1)
        if self.use_sigmoid_cls:
            cls_score = cls_score.reshape(batch_size, -1)
            scores = cls_score.sigmoid()
        else:
            cls_score = cls_score.reshape(batch_size, -1, 2)
            # We set FG labels to [0, num_class-1] and BG label to
            # num_class in RPN head since mmdet v2.5, which is unified to
            # be consistent with other head since mmdet v2.0. In mmdet v2.0
            # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
            scores = cls_score.softmax(-1)[..., 0]
        scores = scores.reshape(batch_size, -1, 1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 6)

        # use static anchor if input shape is static
        if not is_dynamic_flag:
            anchors = anchors.data

        anchors = anchors.unsqueeze(0)

        # topk in tensorrt does not support shape<k
        # concate zero to enable topk,
        scores = pad_with_value_if_necessary(scores, 1, pre_topk, 0.)
        bbox_pred = pad_with_value_if_necessary(bbox_pred, 1, pre_topk)
        anchors = pad_with_value_if_necessary(anchors, 1, pre_topk)

        if pre_topk > 0:
            _, topk_inds = scores.squeeze(2).topk(pre_topk)
            batch_inds = torch.arange(batch_size, device=device).unsqueeze(-1)
            prior_inds = topk_inds.new_zeros((1, 1))
            anchors = anchors[prior_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]
        mlvl_valid_bboxes.append(bbox_pred)
        mlvl_scores.append(scores)
        mlvl_valid_anchors.append(anchors)

    batch_mlvl_bboxes = torch.cat(mlvl_valid_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
    batch_mlvl_bboxes = self.bbox_coder.decode(
        batch_mlvl_anchors,
        batch_mlvl_bboxes,
        max_shape=img_metas[0]['img_shape'])
    # ignore background class
    if not self.use_sigmoid_cls:
        batch_mlvl_scores = batch_mlvl_scores[..., :self.num_classes]
    if not with_nms:
        return batch_mlvl_bboxes, batch_mlvl_scores

    post_params = get_post_processing_params(deploy_cfg)
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    # only one class in rpn
    max_output_boxes_per_class = keep_top_k
    return fake_multiclass_nms_rotated(
        batch_mlvl_bboxes,
        batch_mlvl_scores,
        max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        keep_top_k=keep_top_k,
        version=self.version)
