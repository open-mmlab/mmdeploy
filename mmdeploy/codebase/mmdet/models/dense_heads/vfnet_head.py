import torch

from mmdeploy.codebase.mmdet import (distance2bbox, get_post_processing_params,
                                     multiclass_nms)
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.VFNetHead.get_bboxes')
def vfnet_head__get_bboxes(ctx,
                           self,
                           cls_scores,
                           bbox_preds,
                           bbox_preds_refine,
                           img_metas,
                           cfg=None,
                           rescale=None,
                           with_nms=True):
    """Rewrite `get_bboxes` of `VFNetHead` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

    Args:
        cls_scores (list[Tensor]): Box iou-aware scores for each scale
            level with shape (N, num_points * num_classes, H, W).
        bbox_preds (list[Tensor]): Box offsets for each scale
            level with shape (N, num_points * 4, H, W).
        bbox_preds_refine (list[Tensor]): Refined Box offsets for
            each scale level with shape (N, num_points * 4, H, W).
        img_metas (dict): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        cfg (mmcv.Config): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        with_nms (bool): If True, do nms before returning boxes.
            Default: True.

    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor]: batch_mlvl_bboxes, batch_mlvl_scores
    """
    assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine), \
        'The lengths of lists "cls_scores", "bbox_preds", "bbox_preds_refine"'\
        ' should be the same.'
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    points_list = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                  bbox_preds[0].device)

    cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_pred_list = [bbox_preds_refine[i].detach() for i in range(num_levels)]

    cfg = self.test_cfg if cfg is None else cfg
    batch_size = cls_scores[0].shape[0]
    pre_topk = cfg.get('nms_pre', -1)

    # loop over features, decode boxes
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_points = []
    for cls_score, bbox_pred, points, in zip(cls_score_list, bbox_pred_list,
                                             points_list):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:], \
            'The Height and Width should be the same.'
        scores = cls_score.permute(0, 2, 3,
                                   1).reshape(batch_size, -1,
                                              self.cls_out_channels).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        points = points.expand(batch_size, -1, 2)

        if pre_topk > 0:
            max_scores, _ = scores.max(-1)
            _, topk_inds = max_scores.topk(pre_topk)
            batch_inds = torch.arange(batch_size).view(-1,
                                                       1).expand_as(topk_inds)
            points = points[batch_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]

        mlvl_bboxes.append(bbox_pred)
        mlvl_scores.append(scores)
        mlvl_points.append(points)

    batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    batch_mlvl_points = torch.cat(mlvl_points, dim=1)
    batch_mlvl_bboxes = distance2bbox(
        batch_mlvl_points, batch_mlvl_bboxes, max_shape=img_metas['img_shape'])

    if rescale:
        batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
            img_metas['scale_factor'])

    if not with_nms:
        return batch_mlvl_bboxes, batch_mlvl_scores

    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = cfg.get('nms_pre', post_params.pre_top_k)
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    return multiclass_nms(batch_mlvl_bboxes, batch_mlvl_scores,
                          max_output_boxes_per_class, iou_threshold,
                          score_threshold, pre_top_k, keep_top_k)
