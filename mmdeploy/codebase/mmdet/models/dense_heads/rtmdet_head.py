# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops import multiclass_nms
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.rtmdet_head.'
    'RTMDetHead.predict_by_feat')
def rtmdet_head__predict_by_feat(self,
                                 cls_scores: List[Tensor],
                                 bbox_preds: List[Tensor],
                                 batch_img_metas: Optional[List[dict]] = None,
                                 cfg: Optional[ConfigDict] = None,
                                 rescale: bool = False,
                                 with_nms: bool = True) -> List[InstanceData]:
    """Rewrite `predict_by_feat` of `RTMDet` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        ctx: Context that contains original meta information.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
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
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """

    @mark('rtmdet_head', inputs=['cls_scores', 'bbox_preds'])
    def __mark_pred_maps(cls_scores, bbox_preds):
        return cls_scores, bbox_preds

    cls_scores, bbox_preds = __mark_pred_maps(cls_scores, bbox_preds)
    ctx = FUNCTION_REWRITER.get_context()
    assert len(cls_scores) == len(bbox_preds)
    device = cls_scores[0].device
    cfg = self.test_cfg if cfg is None else cfg
    batch_size = bbox_preds[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, device=device)

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                              self.cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    priors = torch.cat(mlvl_priors)
    tl_x = (priors[..., 0] - flatten_bbox_preds[..., 0])
    tl_y = (priors[..., 1] - flatten_bbox_preds[..., 1])
    br_x = (priors[..., 0] + flatten_bbox_preds[..., 2])
    br_y = (priors[..., 1] + flatten_bbox_preds[..., 3])
    bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    scores = flatten_cls_scores
    if not with_nms:
        return bboxes, scores

    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    nms_type = cfg.nms.get('type')
    return multiclass_nms(
        bboxes,
        scores,
        max_output_boxes_per_class,
        nms_type=nms_type,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.rtmdet_head.'
    'RTMDetHead.predict_by_feat',
    backend=Backend.NCNN.value)
def rtmdet_head__predict_by_feat__ncnn(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True):
    """Rewrite `predict_by_feat` of RTMDetHead for ncnn backend.
    1. Decode the prior to a box format for ncnn DetectionOutput layer to do
    the post-processing.
    2. Batch dimension is not supported by ncnn, but supported by pytorch.
    The negative value of axis in torch.cat is rewritten as corresponding
    positive value to avoid axis shift.
    3. 2-dimension tensor broadcast of `BinaryOps` operator is not supported by
    ncnn. This function unsqueeze 2-dimension tensor to 3-dimension tensor for
    correct `BinaryOps` calculation by ncnn.
    Args:
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        objectnesses (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, 1, H, W).
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
    from mmdeploy.codebase.mmdet.ops import ncnn_detection_output_forward
    from mmdeploy.utils import get_root_logger
    from mmdeploy.utils.config_utils import is_dynamic_shape
    dynamic_flag = is_dynamic_shape(ctx.cfg)
    if dynamic_flag:
        logger = get_root_logger()
        logger.warning('RTMDet does not support dynamic shape with ncnn.')
    img_height = int(batch_img_metas[0]['img_shape'][0])
    img_width = int(batch_img_metas[0]['img_shape'][1])

    assert len(cls_scores) == len(bbox_preds)
    device = cls_scores[0].device
    cfg = self.test_cfg if cfg is None else cfg
    batch_size = bbox_preds[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, device=device, with_stride=True)
    mlvl_priors = [mlvl_prior.unsqueeze(0) for mlvl_prior in mlvl_priors]
    flatten_priors = torch.cat(mlvl_priors, dim=1)

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                              self.cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for bbox_pred in bbox_preds
    ]

    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    dummy_cls_scores = torch.zeros(
        batch_size, cls_scores.shape[-2], 1, device=cls_scores.device)

    batch_mlvl_scores = torch.cat([dummy_cls_scores, cls_scores], dim=2)

    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    assert flatten_priors.shape[-1] == 4, f'rtmdet needs (B, N, 4) priors, got\
        (B, N, {flatten_priors.shape[-1]})'

    tl_x = (flatten_priors[:, :, 0:1] -
            flatten_bbox_preds[:, :, 0:1]) / img_width
    tl_y = (flatten_priors[:, :, 1:2] -
            flatten_bbox_preds[:, :, 1:2]) / img_height
    br_x = (flatten_priors[:, :, 0:1] +
            flatten_bbox_preds[:, :, 2:3]) / img_width
    br_y = (flatten_priors[:, :, 1:2] +
            flatten_bbox_preds[:, :, 3:4]) / img_height
    prior_box_ncnn = torch.stack([tl_x, tl_y, br_x, br_y], -1)

    scores = batch_mlvl_scores

    batch_mlvl_bboxes = flatten_bbox_preds.reshape(batch_size, 1, -1)
    batch_mlvl_scores = scores.reshape(batch_size, 1, -1)
    batch_mlvl_priors = prior_box_ncnn.reshape(batch_size, 1, -1)
    batch_mlvl_vars = torch.ones_like(batch_mlvl_priors)
    batch_mlvl_priors = torch.cat([batch_mlvl_priors, batch_mlvl_vars], dim=1)
    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    vars = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    output__ncnn = ncnn_detection_output_forward(
        batch_mlvl_bboxes, batch_mlvl_scores, batch_mlvl_priors,
        score_threshold, iou_threshold, pre_top_k, keep_top_k,
        self.num_classes + 1,
        vars.cpu().detach().numpy())
    return output__ncnn
