# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmdet.models.dense_heads import PAAHead
from mmdet.models.task_modules.coders import (DeltaXYWHBBoxCoder,
                                              DistancePointBBoxCoder,
                                              TBLRBBoxCoder)
from mmdet.structures.bbox import BaseBoxes, get_box_tensor
from mmdet.structures.bbox.transforms import distance2bbox
from mmengine import ConfigDict
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import (gather_topk,
                                            get_post_processing_params,
                                            pad_with_value_if_necessary)
from mmdeploy.codebase.mmdet.ops import ncnn_detection_output_forward
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops import multiclass_nms
from mmdeploy.utils import Backend, is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.base_dense_head.'
    'BaseDenseHead.predict_by_feat')
def base_dense_head__predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        score_factors: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
        **kwargs):
    """Rewrite `predict_by_feat` of `BaseDenseHead` for default backend.
    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.
    Args:
        ctx (ContextCaller): The context with additional information.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        score_factors (list[Tensor], optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W). Defaults to None.
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.
    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor, Tensor]: batch_mlvl_bboxes,
                batch_mlvl_scores, batch_mlvl_centerness
    """
    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

    # anchor could be subclass of BaseBoxes in mmrotate
    prior_type = type(mlvl_priors[0])
    mlvl_priors = [get_box_tensor(priors) for priors in mlvl_priors]

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
    assert batch_img_metas is not None
    img_shape = batch_img_metas[0]['img_shape']

    assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
    batch_size = cls_scores[0].shape[0]
    cfg = self.test_cfg
    pre_topk = cfg.get('nms_pre', -1)

    mlvl_valid_bboxes = []
    mlvl_valid_scores = []
    mlvl_valid_priors = []

    for cls_score, bbox_pred, score_factors, priors in zip(
            mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor, mlvl_priors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                       self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(-1)[:, :, :-1]
        if with_score_factors:
            score_factors = score_factors.permute(0, 2, 3,
                                                  1).reshape(batch_size,
                                                             -1).sigmoid()
            score_factors = score_factors.unsqueeze(2)
        dim = self.bbox_coder.encode_size
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, dim)
        if not is_dynamic_flag:
            priors = priors.data

        if pre_topk > 0:
            priors = pad_with_value_if_necessary(priors, 1, pre_topk)
            bbox_pred = pad_with_value_if_necessary(bbox_pred, 1, pre_topk)
            scores = pad_with_value_if_necessary(scores, 1, pre_topk, 0.)
            if with_score_factors:
                score_factors = pad_with_value_if_necessary(
                    score_factors, 1, pre_topk, 0.)

            nms_pre_score = scores
            if with_score_factors:
                nms_pre_score = nms_pre_score * score_factors
                if isinstance(self, PAAHead):
                    nms_pre_score = nms_pre_score.sqrt()

            # Get maximum scores for foreground classes.
            if self.use_sigmoid_cls:
                max_scores, _ = nms_pre_score.max(-1)
            else:
                max_scores, _ = nms_pre_score[..., :-1].max(-1)
            _, topk_inds = max_scores.topk(pre_topk)
            bbox_pred, scores, score_factors = gather_topk(
                bbox_pred,
                scores,
                score_factors,
                inds=topk_inds,
                batch_size=batch_size,
                is_batched=True)
            priors = gather_topk(
                priors,
                inds=topk_inds,
                batch_size=batch_size,
                is_batched=False)

        mlvl_valid_bboxes.append(bbox_pred)
        mlvl_valid_scores.append(scores)
        mlvl_valid_priors.append(priors)
        if with_score_factors:
            mlvl_score_factors.append(score_factors)

    batch_mlvl_bboxes_pred = torch.cat(mlvl_valid_bboxes, dim=1)
    batch_scores = torch.cat(mlvl_valid_scores, dim=1)
    batch_priors = torch.cat(mlvl_valid_priors, dim=1)

    if issubclass(prior_type, BaseBoxes):
        batch_priors = prior_type(batch_priors, clone=False)

    batch_bboxes = self.bbox_coder.decode(
        batch_priors, batch_mlvl_bboxes_pred, max_shape=img_shape)

    batch_bboxes = get_box_tensor(batch_bboxes)

    if with_score_factors:
        batch_score_factors = torch.cat(mlvl_score_factors, dim=1)
    if not self.use_sigmoid_cls:
        batch_scores = batch_scores[..., :self.num_classes]

    if with_score_factors:
        batch_scores = batch_scores * batch_score_factors
        if isinstance(self, PAAHead):
            batch_scores = batch_scores.sqrt()

    if not with_nms:
        return batch_bboxes, batch_scores

    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    nms_type = cfg.nms.get('type')
    return multiclass_nms(
        batch_bboxes,
        batch_scores,
        max_output_boxes_per_class,
        nms_type=nms_type,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.base_dense_head.'
    'BaseDenseHead.predict_by_feat',
    backend=Backend.RKNN.value)
def base_dense_head__predict_by_feat__rknn(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        score_factors: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
        **kwargs):
    """Rewrite `predict_by_feat` of `BaseDenseHead` for default backend.
    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.
    Args:
        ctx (ContextCaller): The context with additional information.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        score_factors (list[Tensor], optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W). Defaults to None.
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.
    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor, Tensor]: batch_mlvl_bboxes,
                batch_mlvl_scores, batch_mlvl_centerness
    """
    ctx = FUNCTION_REWRITER.get_context()

    # mark nodes for partition
    @mark('BaseDenseHead', outputs=['BaseDenseHead.cls', 'BaseDenseHead.loc'])
    def __mark_dense_head(cls_scores, bbox_preds):
        return cls_scores, bbox_preds

    cls_scores, bbox_preds = __mark_dense_head(cls_scores, bbox_preds)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
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
    assert batch_img_metas is not None
    img_shape = batch_img_metas[0]['img_shape']

    assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
    batch_size = cls_scores[0].shape[0]

    mlvl_valid_bboxes = []
    mlvl_valid_scores = []
    mlvl_valid_priors = []

    for cls_score, bbox_pred, score_factors, priors in zip(
            mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor, mlvl_priors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                       self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(-1)[:, :, :-1]
        if with_score_factors:
            score_factors = score_factors.permute(0, 2, 3,
                                                  1).reshape(batch_size,
                                                             -1).sigmoid()
            score_factors = score_factors.unsqueeze(2)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        if not is_dynamic_flag:
            priors = priors.data

        mlvl_valid_bboxes.append(bbox_pred)
        mlvl_valid_scores.append(scores)
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
        if isinstance(self, PAAHead):
            batch_scores = batch_scores.sqrt()
    return batch_bboxes, batch_scores


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.base_dense_head.'
    'BaseDenseHead.predict_by_feat',
    backend=Backend.NCNN.value)
def base_dense_head__predict_by_feat__ncnn(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        score_factors: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
        **kwargs):
    """Rewrite `predict_by_feat` of BaseDenseHead for ncnn backend.

    Shape node and batch inference is not supported by ncnn. This function
    transform dynamic shape to constant shape and remove batch inference.

    Args:
        ctx (ContextCaller): The context with additional information.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        score_factors (list[Tensor], optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W). Defaults to None.
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        output__ncnn (Tensor): outputs, shape is [N, num_det, 6].
    """
    ctx = FUNCTION_REWRITER.get_context()
    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = ctx.cfg
    assert not is_dynamic_shape(deploy_cfg), 'base_dense_head for ncnn\
        only supports static shape.'

    if score_factors is None:
        # e.g. Retina, FreeAnchor, Foveabox, etc.
        with_score_factors = False
    else:
        # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
        with_score_factors = True
        assert len(cls_scores) == len(score_factors)
    batch_size = cls_scores[0].shape[0]
    assert batch_size == 1, f'ncnn deployment requires batch size 1, \
        got {batch_size}.'

    num_levels = len(cls_scores)
    if with_score_factors:
        score_factor_list = score_factors
    else:
        score_factor_list = [None for _ in range(num_levels)]

    if isinstance(self.bbox_coder, DeltaXYWHBBoxCoder):
        vars = torch.tensor(self.bbox_coder.stds)
    elif isinstance(self.bbox_coder, TBLRBBoxCoder):
        normalizer = self.bbox_coder.normalizer
        if isinstance(normalizer, float):
            vars = torch.tensor([normalizer, normalizer, 1, 1],
                                dtype=torch.float32)
        else:
            assert len(normalizer) == 4, f'normalizer of tblr must be 4,\
                got {len(normalizer)}'

            assert (normalizer[0] == normalizer[1] and normalizer[2]
                    == normalizer[3]), 'normalizer between top \
                        and bottom, left and right must be the same value, or \
                        we can not transform it to delta_xywh format.'

            vars = torch.tensor([normalizer[0], normalizer[2], 1, 1],
                                dtype=torch.float32)
    elif isinstance(self.bbox_coder, DistancePointBBoxCoder):
        vars = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    else:
        vars = torch.tensor([1, 1, 1, 1], dtype=torch.float32)

    if isinstance(batch_img_metas[0]['img_shape'][0], int):
        assert isinstance(batch_img_metas[0]['img_shape'][1], int)
        img_height = batch_img_metas[0]['img_shape'][0]
        img_width = batch_img_metas[0]['img_shape'][1]
    else:
        img_height = batch_img_metas[0]['img_shape'][0].item()
        img_width = batch_img_metas[0]['img_shape'][1].item()
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, device=cls_scores[0].device)
    batch_mlvl_priors = []
    for i in range(num_levels):
        _priors = mlvl_priors[i].reshape(1, -1, mlvl_priors[i].shape[-1])
        x1 = _priors[:, :, 0:1] / img_width
        y1 = _priors[:, :, 1:2] / img_height
        x2 = _priors[:, :, 2:3] / img_width
        y2 = _priors[:, :, 3:4] / img_height
        priors = torch.cat([x1, y1, x2, y2], dim=2).data
        batch_mlvl_priors.append(priors)

    cfg = self.test_cfg if cfg is None else cfg

    batch_mlvl_bboxes = []
    batch_mlvl_scores = []
    batch_mlvl_score_factors = []

    for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
            enumerate(zip(cls_scores, bbox_preds,
                      score_factor_list, batch_mlvl_priors)):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        # ncnn needs 3 dimensions to reshape when including -1 parameter in
        # width or height dimension.
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        if with_score_factors:
            score_factor = score_factor.permute(0, 2, 3, 1).\
                reshape(batch_size, -1, 1).sigmoid()
        cls_score = cls_score.permute(0, 2, 3, 1).\
            reshape(batch_size, -1, self.cls_out_channels)
        # ncnn DetectionOutput op needs num_class + 1 classes. So if sigmoid
        # score, we should padding background class according to mmdetection
        # num_class definition.
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
            dummy_background_score = torch.zeros(
                batch_size, cls_score.shape[1], 1, device=cls_score.device)
            scores = torch.cat([scores, dummy_background_score], dim=2)
        else:
            scores = cls_score.softmax(-1)
        batch_mlvl_bboxes.append(bbox_pred)
        batch_mlvl_scores.append(scores)
        batch_mlvl_score_factors.append(score_factor)

    batch_mlvl_priors = torch.cat(batch_mlvl_priors, dim=1)
    batch_mlvl_scores = torch.cat(batch_mlvl_scores, dim=1)
    batch_mlvl_bboxes = torch.cat(batch_mlvl_bboxes, dim=1)
    batch_mlvl_scores = torch.cat([
        batch_mlvl_scores[:, :, self.num_classes:],
        batch_mlvl_scores[:, :, 0:self.num_classes]
    ],
                                  dim=2)
    if isinstance(self.bbox_coder, TBLRBBoxCoder):
        batch_mlvl_bboxes = _tblr_pred_to_delta_xywh_pred(
            batch_mlvl_bboxes, vars[0:2])
    elif isinstance(self.bbox_coder, DistancePointBBoxCoder):
        bboxes_x0 = batch_mlvl_bboxes[:, :, 0:1] / img_width
        bboxes_y0 = batch_mlvl_bboxes[:, :, 1:2] / img_height
        bboxes_x1 = batch_mlvl_bboxes[:, :, 2:3] / img_width
        bboxes_y1 = batch_mlvl_bboxes[:, :, 3:4] / img_height
        batch_mlvl_bboxes = torch.cat(
            [bboxes_x0, bboxes_y0, bboxes_x1, bboxes_y1], dim=2)
        batch_mlvl_priors = distance2bbox(batch_mlvl_priors, batch_mlvl_bboxes)

    if with_score_factors:
        batch_mlvl_score_factors = torch.cat(batch_mlvl_score_factors, dim=1)
        batch_mlvl_scores = batch_mlvl_scores.permute(
            0, 2, 1).unsqueeze(3) * batch_mlvl_score_factors.permute(
                0, 2, 1).unsqueeze(3)
        batch_mlvl_scores = batch_mlvl_scores.squeeze(3).permute(0, 2, 1)
        if isinstance(self, PAAHead):
            batch_mlvl_scores = batch_mlvl_scores.sqrt()

    # flatten for ncnn DetectionOutput op inputs.
    batch_mlvl_vars = vars.expand_as(batch_mlvl_priors)
    batch_mlvl_bboxes = batch_mlvl_bboxes.reshape(batch_size, 1, -1)
    batch_mlvl_scores = batch_mlvl_scores.reshape(batch_size, 1, -1)
    batch_mlvl_priors = batch_mlvl_priors.reshape(batch_size, 1, -1)
    batch_mlvl_vars = batch_mlvl_vars.reshape(batch_size, 1, -1)
    batch_mlvl_priors = torch.cat([batch_mlvl_priors, batch_mlvl_vars], dim=1)
    if not isinstance(self.bbox_coder, DistancePointBBoxCoder):
        batch_mlvl_priors = batch_mlvl_priors.data

    post_params = get_post_processing_params(ctx.cfg)
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    output__ncnn = ncnn_detection_output_forward(
        batch_mlvl_bboxes, batch_mlvl_scores, batch_mlvl_priors,
        score_threshold, iou_threshold, pre_top_k, keep_top_k,
        self.num_classes + 1,
        vars.cpu().detach().numpy())

    return output__ncnn


def _tblr_pred_to_delta_xywh_pred(bbox_pred: torch.Tensor,
                                  normalizer: torch.Tensor) -> torch.Tensor:
    """Transform tblr format bbox prediction to delta_xywh format for ncnn.

    An internal function for transforming tblr format bbox prediction to
    delta_xywh format. ncnn DetectionOutput layer needs delta_xywh format
    bbox_pred as input.

    Args:
        bbox_pred (Tensor): The bbox prediction of tblr format, has shape
            (N, num_det, 4).
        normalizer (Tensor): The normalizer scale of bbox horizon and
            vertical coordinates, has shape (2,).

    Returns:
        Tensor: The delta_xywh format bbox predictions.
    """
    top = bbox_pred[:, :, 0:1]
    bottom = bbox_pred[:, :, 1:2]
    left = bbox_pred[:, :, 2:3]
    right = bbox_pred[:, :, 3:4]
    h = (top + bottom) * normalizer[0]
    w = (left + right) * normalizer[1]

    _dwh = torch.cat([w, h], dim=2)
    assert torch.all(_dwh >= 0), 'wh must be positive before log.'
    dwh = torch.log(_dwh)

    return torch.cat([(right - left) / 2, (bottom - top) / 2, dwh], dim=2)
