# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet import (get_post_processing_params,
                                     pad_with_value_if_necessary)
from mmdeploy.codebase.mmrotate.core.post_processing import \
    multiclass_nms_rotated
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmrotate.models.dense_heads.rotated_anchor_head.'
    'RotatedAnchorHead.get_bboxes')
def rotated_anchor_head__get_bbox(ctx,
                                  self,
                                  cls_scores,
                                  bbox_preds,
                                  img_metas=None,
                                  cfg=None,
                                  rescale=False,
                                  with_nms=True,
                                  **kwargs):
    """Rewrite `get_bboxes` of `RotatedAnchorHead` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            , 1., num_priors * 5, H, W).
        img_metas (list[dict], Optional): Image meta info. Default None.
        cfg (mmcv.Config, Optional): Test / postprocessing configuration,
            if None, test_cfg would be used.  Default None.
        rescale (bool): If True, return boxes in original image space.
            Default False.
        with_nms (bool): If True, do nms before return boxes.
            Default True.

    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor, Tensor]: batch_mlvl_bboxes,
                batch_mlvl_scores, batch_mlvl_centerness
    """
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_priors = self.anchor_generator.grid_priors(
        featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

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
    mlvl_valid_priors = []

    for cls_score, bbox_pred, priors in zip(mlvl_cls_scores, mlvl_bbox_preds,
                                            mlvl_priors):

        scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                       self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(-1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 5)
        if not is_dynamic_flag:
            priors = priors.data
        priors = priors.expand(batch_size, -1, priors.size(-1))
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
            batch_inds = torch.arange(
                batch_size,
                device=bbox_pred.device).view(-1, 1).expand_as(topk_inds)
            priors = priors[batch_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]

        mlvl_valid_bboxes.append(bbox_pred)
        mlvl_valid_scores.append(scores)
        mlvl_valid_priors.append(priors)

    batch_mlvl_bboxes_pred = torch.cat(mlvl_valid_bboxes, dim=1)
    batch_scores = torch.cat(mlvl_valid_scores, dim=1)
    batch_priors = torch.cat(mlvl_valid_priors, dim=1)
    batch_bboxes = self.bbox_coder.decode(
        batch_priors, batch_mlvl_bboxes_pred, max_shape=img_shape)

    if not self.use_sigmoid_cls:
        batch_scores = batch_scores[..., :self.num_classes]

    if not with_nms:
        return batch_bboxes, batch_scores

    post_params = get_post_processing_params(deploy_cfg)
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    return multiclass_nms_rotated(
        batch_bboxes,
        batch_scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)
