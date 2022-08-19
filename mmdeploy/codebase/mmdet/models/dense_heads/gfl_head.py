# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdeploy.codebase.mmdet import (get_post_processing_params,
                                     multiclass_nms, pad_with_value)
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend, get_backend, is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.gfl_head.'
    'GFLHead.get_bboxes')
def gfl_head__get_bbox(ctx,
                       self,
                       cls_scores,
                       bbox_preds,
                       score_factors=None,
                       img_metas=None,
                       cfg=None,
                       rescale=False,
                       with_nms=True,
                       **kwargs):
    """Rewrite `get_bboxes` of `GFLHead` for default backend.

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
            (batch_size, num_priors * 4, H, W).
        score_factors (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W). Default None.
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
    backend = get_backend(deploy_cfg)
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
    mlvl_priors = [priors.unsqueeze(0) for priors in mlvl_priors]

    mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    if score_factors is None:
        with_score_factors = False
        mlvl_score_factor = [None for _ in range(num_levels)]
    else:
        with_score_factors = True
        mlvl_score_factor = [
            score_factors[i].detach() for i in range(num_levels)
        ]
        mlvl_score_factors = []
    assert img_metas is not None
    img_shape = img_metas[0]['img_shape']

    assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
    batch_size = cls_scores[0].shape[0]
    cfg = self.test_cfg
    pre_topk = cfg.get('nms_pre', -1)

    mlvl_valid_bboxes = []
    mlvl_valid_scores = []
    mlvl_valid_priors = []

    for cls_score, bbox_pred, score_factors, priors, stride in zip(
            mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor, mlvl_priors,
            self.prior_generator.strides):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        assert stride[0] == stride[1]

        scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                       self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = scores.sigmoid()
            nms_pre_score = scores
        else:
            scores = scores.softmax(-1)
            nms_pre_score = scores
        if with_score_factors:
            score_factors = score_factors.permute(0, 2, 3,
                                                  1).reshape(batch_size,
                                                             -1).sigmoid()
            score_factors = score_factors.unsqueeze(2)
        bbox_pred = batched_integral(self.integral,
                                     bbox_pred.permute(0, 2, 3, 1)) * stride[0]
        if not is_dynamic_flag:
            priors = priors.data
        if pre_topk > 0:
            if with_score_factors:
                nms_pre_score = nms_pre_score * score_factors
            if backend == Backend.TENSORRT:
                priors = pad_with_value(priors, 1, pre_topk)
                bbox_pred = pad_with_value(bbox_pred, 1, pre_topk)
                scores = pad_with_value(scores, 1, pre_topk, 0.)
                nms_pre_score = pad_with_value(nms_pre_score, 1, pre_topk, 0.)
                if with_score_factors:
                    score_factors = pad_with_value(score_factors, 1, pre_topk,
                                                   0.)

            # Get maximum scores for foreground classes.
            if self.use_sigmoid_cls:
                max_scores, _ = nms_pre_score.max(-1)
            else:
                max_scores, _ = nms_pre_score[..., :-1].max(-1)
            _, topk_inds = max_scores.topk(pre_topk)
            batch_inds = torch.arange(
                batch_size, device=bbox_pred.device).unsqueeze(-1)
            prior_inds = batch_inds.new_zeros((1, 1))
            priors = priors[prior_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]
            if with_score_factors:
                score_factors = score_factors[batch_inds, topk_inds, :]

        mlvl_valid_bboxes.append(bbox_pred)
        mlvl_valid_scores.append(scores)
        priors = self.anchor_center(priors)
        mlvl_valid_priors.append(priors)
        if with_score_factors:
            mlvl_score_factors.append(score_factors)

    batch_mlvl_bboxes_pred = torch.cat(mlvl_valid_bboxes, dim=1)
    batch_scores = torch.cat(mlvl_valid_scores, dim=1)
    batch_priors = torch.cat(mlvl_valid_priors, dim=1)
    batch_bboxes = self.bbox_coder.decode(
        batch_priors, batch_mlvl_bboxes_pred, max_shape=img_shape)
    if with_score_factors:
        batch_score_factors = torch.cat(mlvl_score_factors, dim=1)

    if not self.use_sigmoid_cls:
        batch_scores = batch_scores[..., :self.num_classes]

    if with_score_factors:
        batch_scores = batch_scores * batch_score_factors
    if not with_nms:
        return batch_bboxes, batch_scores
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


def batched_integral(intergral, x):
    batch_size = x.size(0)
    x = F.softmax(x.reshape(batch_size, -1, intergral.reg_max + 1), dim=2)
    x = F.linear(x,
                 intergral.project.type_as(x).unsqueeze(0)).reshape(
                     batch_size, -1, 4)
    return x
