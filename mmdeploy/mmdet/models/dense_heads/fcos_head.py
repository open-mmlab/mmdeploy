import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmdet.core import distance2bbox, multiclass_nms
from mmdeploy.mmdet.export import pad_with_value
from mmdeploy.utils import (Backend, get_backend, get_mmdet_params,
                            is_dynamic_shape)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.FCOSHead.get_bboxes')
def fcos_head__get_bboxes(ctx,
                          self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          img_metas,
                          with_nms=True,
                          cfg=None,
                          **kwargs):
    """Rewrite `get_bboxes` of FCOSHead for default backend.

    Rewrite this function to support deployment of default backend
    and dynamic shape export. Transform network output for a batch into
    bbox predictions.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (ATSSHead): The instance of the class ATSSHead.
        cls_scores (list[Tensor]): Box scores for each scale level
            with shape (N, num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W).
        centernesses (list[Tensor]): Centerness for each scale level with
            shape (N, num_anchors * 1, H, W).
        img_metas (dict): Meta information of the image, e.g.,
            image size, scaling factor, etc.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.


    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor, Tensor]: batch_mlvl_bboxes,
                batch_mlvl_scores, batch_mlvl_centerness
    """

    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    points_list = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                  bbox_preds[0].device)

    cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
    centerness_pred_list = [
        centernesses[i].detach() for i in range(num_levels)
    ]

    cfg = self.test_cfg if cfg is None else cfg
    assert len(cls_scores) == len(bbox_preds) == len(points_list)
    batch_size = cls_scores[0].shape[0]
    pre_topk = cfg.get('nms_pre', -1)

    # loop over features, decode boxes
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_centerness = []
    mlvl_points = []
    for level_id, cls_score, bbox_pred, centerness, points in zip(
            range(num_levels), cls_score_list, bbox_pred_list,
            centerness_pred_list, points_list):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        scores = cls_score.permute(0, 2, 3,
                                   1).reshape(batch_size, -1,
                                              self.cls_out_channels).sigmoid()
        centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                            1).sigmoid()

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        # use static anchor if input shape is static
        if not is_dynamic_flag:
            points = points.data

        points = points.expand(batch_size, -1, 2)

        backend = get_backend(deploy_cfg)
        # topk in tensorrt does not support shape<k
        # concate zero to enable topk,
        if backend == Backend.TENSORRT:
            scores = pad_with_value(scores, 1, pre_topk, 0.)
            centerness = pad_with_value(centerness, 1, pre_topk)
            bbox_pred = pad_with_value(bbox_pred, 1, pre_topk)
            points = pad_with_value(points, 1, pre_topk)

        if pre_topk > 0:
            max_scores, _ = (scores * centerness).max(-1)
            _, topk_inds = max_scores.topk(pre_topk)
            batch_inds = torch.arange(batch_size).view(-1,
                                                       1).expand_as(topk_inds)

            points = points[batch_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]
            centerness = centerness[batch_inds, topk_inds, :]

        mlvl_points.append(points)
        mlvl_bboxes.append(bbox_pred)
        mlvl_scores.append(scores)
        mlvl_centerness.append(centerness)

    batch_mlvl_points = torch.cat(mlvl_points, dim=1)
    batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
    batch_mlvl_bboxes = distance2bbox(
        batch_mlvl_points, batch_mlvl_bboxes, max_shape=img_metas['img_shape'])

    if not with_nms:
        return batch_mlvl_bboxes, batch_mlvl_scores, batch_mlvl_centerness

    batch_mlvl_scores = batch_mlvl_scores * batch_mlvl_centerness
    post_params = get_mmdet_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    return multiclass_nms(batch_mlvl_bboxes, batch_mlvl_scores,
                          max_output_boxes_per_class, iou_threshold,
                          score_threshold, pre_top_k, keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.FCOSHead.get_bboxes', backend='ncnn')
def fcos_head__get_bboxes__ncnn(ctx,
                                self,
                                cls_scores,
                                bbox_preds,
                                centernesses,
                                img_metas,
                                with_nms=True,
                                cfg=None,
                                **kwargs):
    """Rewrite `get_bboxes` of FCOSHead for ncnn backend.

    1. Shape node and batch inference is not supported by ncnn. This function
    transform dynamic shape to constant shape and remove batch inference.
    2. 2-dimension tensor broadcast of `BinaryOps` operator is not supported by
    ncnn. This function unsqueeze 2-dimension tensor to 3-dimension tensor for
    correct `BinaryOps` calculation by ncnn.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (ATSSHead): The instance of the class ATSSHead.
        cls_scores (list[Tensor]): Box scores for each scale level
            with shape (N, num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 4, H, W).
        centernesses (list[Tensor]): Centerness for each scale level with
            shape (N, num_anchors * 1, H, W).
        img_metas (dict): Meta information of the image, e.g.,
            image size, scaling factor, etc.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.


    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor, Tensor]: batch_mlvl_bboxes,
                batch_mlvl_scores, batch_mlvl_centerness
    """
    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = ctx.cfg
    assert not is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    points_list = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                  bbox_preds[0].device)

    cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
    centerness_pred_list = [
        centernesses[i].detach() for i in range(num_levels)
    ]

    cfg = self.test_cfg if cfg is None else cfg
    assert len(cls_scores) == len(bbox_preds) == len(points_list)
    batch_size = 1
    pre_topk = cfg.get('nms_pre', -1)

    # loop over features, decode boxes
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_centerness = []
    mlvl_points = []
    for level_id, cls_score, bbox_pred, centerness, points in zip(
            range(num_levels), cls_score_list, bbox_pred_list,
            centerness_pred_list, points_list):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        scores = cls_score.permute(0, 2, 3,
                                   1).reshape(batch_size, -1,
                                              self.cls_out_channels).sigmoid()
        centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                            1).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        points = points.expand(1, -1, 2).data
        if pre_topk > 0:

            _scores = scores.reshape(batch_size, -1, self.cls_out_channels, 1)
            _centerness = centerness.reshape(batch_size, -1, 1, 1)
            max_scores, _ = (_scores * _centerness). \
                reshape(batch_size, -1, self.cls_out_channels).max(-1)

            _, topk_inds = max_scores.topk(pre_topk)

            topk_inds = topk_inds.view(-1)

            points = points[:, topk_inds, :]
            bbox_pred = bbox_pred[:, topk_inds, :]
            scores = scores[:, topk_inds, :]
            centerness = centerness[:, topk_inds, :]
        mlvl_points.append(points)
        mlvl_bboxes.append(bbox_pred)
        mlvl_scores.append(scores)
        mlvl_centerness.append(centerness)

    batch_mlvl_points = torch.cat(mlvl_points, dim=1)
    batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
    batch_mlvl_bboxes = distance2bbox(
        batch_mlvl_points, batch_mlvl_bboxes, max_shape=img_metas['img_shape'])

    if not with_nms:
        return batch_mlvl_bboxes, batch_mlvl_scores, batch_mlvl_centerness

    _batch_mlvl_scores = batch_mlvl_scores.unsqueeze(3)
    _batch_mlvl_centerness = batch_mlvl_centerness.unsqueeze(3)
    batch_mlvl_scores = (_batch_mlvl_scores * _batch_mlvl_centerness). \
        reshape(batch_mlvl_scores.shape)
    batch_mlvl_bboxes = batch_mlvl_bboxes.reshape(batch_size, -1, 4)
    post_params = get_mmdet_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    return multiclass_nms(batch_mlvl_bboxes, batch_mlvl_scores,
                          max_output_boxes_per_class, iou_threshold,
                          score_threshold, pre_top_k, keep_top_k)
