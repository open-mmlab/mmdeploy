# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet.core.post_processing import multiclass_nms
from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER, mark


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.YOLOXHead.get_bboxes')
def yolox_head__get_bboxes(ctx,
                           self,
                           cls_scores,
                           bbox_preds,
                           objectnesses,
                           img_metas=None,
                           cfg=None,
                           rescale=False,
                           with_nms=True):
    """Rewrite `get_bboxes` of `YOLOXHead` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        ctx: Context that contains original meta information.
        self: Represent the instance of the original class.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        objectnesses (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, 1, H, W).
        img_metas (list[dict]): Image meta info. Default None.
        cfg (mmcv.Config, Optional): Test / postprocessing configuration,
            if None, test_cfg would be used.  Default None.
        rescale (bool): If True, return boxes in original image space.
            Default False.
        with_nms (bool): If True, do nms before return boxes.
            Default True.

    Returns:
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """
    # mark pred_maps
    @mark('yolo_head', inputs=['cls_scores', 'bbox_preds', 'objectnesses'])
    def __mark_pred_maps(cls_scores, bbox_preds, objectnesses):
        return cls_scores, bbox_preds, objectnesses

    cls_scores, bbox_preds, objectnesses = __mark_pred_maps(
        cls_scores, bbox_preds, objectnesses)
    assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
    device = cls_scores[0].device
    cfg = self.test_cfg if cfg is None else cfg
    batch_size = bbox_preds[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, device=device, with_stride=True)

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                              self.cls_out_channels)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(batch_size, -1)
        for objectness in objectnesses
    ]

    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    score_factor = torch.cat(flatten_objectness, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_priors = torch.cat(mlvl_priors)
    bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
    # directly multiply score factor and feed to nms
    scores = cls_scores * (score_factor.unsqueeze(-1))

    if not with_nms:
        return bboxes, scores

    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    return multiclass_nms(bboxes, scores, max_output_boxes_per_class,
                          iou_threshold, score_threshold, pre_top_k,
                          keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.YOLOXHead.get_bboxes', backend='ncnn')
def yolox_head__get_bboxes__ncnn(ctx,
                                 self,
                                 cls_scores,
                                 bbox_preds,
                                 objectnesses,
                                 img_metas=None,
                                 cfg=None,
                                 rescale=False,
                                 with_nms=True):
    """Rewrite `get_bboxes` of YOLOXHead for ncnn backend.

    1. Decode the prior to a box format for ncnn DetectionOutput layer to do
    the post-processing.
    2. Batch dimension is not supported by ncnn, but supported by pytorch.
    The negative value of axis in torch.cat is rewritten as corresponding
    positive value to avoid axis shift.
    3. 2-dimension tensor broadcast of `BinaryOps` operator is not supported by
    ncnn. This function unsqueeze 2-dimension tensor to 3-dimension tensor for
    correct `BinaryOps` calculation by ncnn.

    Args:
        ctx: Context that contains original meta information.
        self: Represent the instance of the original class.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        objectnesses (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, 1, H, W).
        img_metas (list[dict]): Image meta info. Default None.
        cfg (mmcv.Config, Optional): Test / postprocessing configuration,
            if None, test_cfg would be used.  Default None.
        rescale (bool): If True, return boxes in original image space.
            Default False.
        with_nms (bool): If True, do nms before return boxes.
            Default True.

    Returns:
        output__ncnn (Tensor): outputs, shape is [N, num_det, 6].
    """
    from mmdeploy.codebase.mmdet.core.ops.detection_output import \
        ncnn_detection_output_forward
    from mmdeploy.utils import get_root_logger
    from mmdeploy.utils.config_utils import is_dynamic_shape
    dynamic_flag = is_dynamic_shape(ctx.cfg)
    if dynamic_flag:
        logger = get_root_logger()
        logger.warning('YOLOX does not support dynamic shape with ncnn.')
    img_height = int(img_metas[0]['img_shape'][0])
    img_width = int(img_metas[0]['img_shape'][1])

    assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
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
    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
        for objectness in objectnesses
    ]

    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    dummy_cls_scores = torch.zeros(
        batch_size, cls_scores.shape[-2], 1, device=cls_scores.device)

    batch_mlvl_scores = torch.cat([dummy_cls_scores, cls_scores], dim=2)
    score_factor = torch.cat(flatten_objectness, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    assert flatten_priors.shape[-1] == 4, \
        'yolox needs (B, N, 4) priors, got ' \
        f'(B, N, {flatten_priors.shape[-1]})'
    prior_box_x1 = (flatten_priors[:, :, 0:1] - flatten_priors[:, :, 2:3] / 2)\
        / img_width
    prior_box_y1 = (flatten_priors[:, :, 1:2] - flatten_priors[:, :, 3:4] / 2)\
        / img_height
    prior_box_x2 = (flatten_priors[:, :, 0:1] + flatten_priors[:, :, 2:3] / 2)\
        / img_width
    prior_box_y2 = (flatten_priors[:, :, 1:2] + flatten_priors[:, :, 3:4] / 2)\
        / img_height
    prior_box_ncnn = torch.cat(
        [prior_box_x1, prior_box_y1, prior_box_x2, prior_box_y2], dim=2)

    scores = batch_mlvl_scores.permute(0, 2, 1).unsqueeze(3) * \
        score_factor.permute(0, 2, 1).unsqueeze(3)
    scores = scores.squeeze(3).permute(0, 2, 1)

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
