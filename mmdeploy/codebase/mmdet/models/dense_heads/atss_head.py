import torch

from mmdeploy.codebase.mmdet import get_post_processing_params, multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.ATSSHead.get_bboxes')
def atss_head__get_bboxes(ctx,
                          self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          img_metas,
                          cfg=None,
                          rescale=False,
                          with_nms=True):
    """Rewrite `get_bboxes` of `ATSSHead` for default backend.

    Rewrite this function to deploy model, transform network output for a
    batch into bbox predictions.

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
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        If with_nms == True:
            tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
        Else:
            tuple[Tensor, Tensor, Tensor]: batch_mlvl_bboxes,
                batch_mlvl_scores, batch_mlvl_centerness
    """
    cfg = self.test_cfg if cfg is None else cfg
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device=cls_scores[0].device)

    cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
    centerness_pred_list = [
        centernesses[i].detach() for i in range(num_levels)
    ]

    img_shape = img_metas['img_shape']
    batch_size = cls_score_list[0].shape[0]
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_centerness = []
    for cls_score, bbox_pred, centerness, anchors in zip(
            cls_score_list, bbox_pred_list, centerness_pred_list,
            mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        scores = cls_score.permute(0, 2, 3,
                                   1).reshape(batch_size, -1,
                                              self.cls_out_channels).sigmoid()
        centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size,
                                                            -1).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        anchors = anchors.expand_as(bbox_pred)
        bboxes = self.bbox_coder.decode(
            anchors, bbox_pred, max_shape=img_shape[:2])
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_centerness.append(centerness)
    batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    if rescale:
        scale_factor = img_metas['scale_factor']
        batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
            scale_factor).unsqueeze(1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

    if not with_nms:
        return batch_mlvl_bboxes, batch_mlvl_scores, batch_mlvl_centerness

    batch_mlvl_centerness = batch_mlvl_centerness.unsqueeze(-1).expand(
        batch_mlvl_scores.shape)
    batch_mlvl_scores = batch_mlvl_scores * batch_mlvl_centerness
    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    nms_pre = cfg.get('deploy_nms_pre', -1)
    det_results = multiclass_nms(batch_mlvl_bboxes, batch_mlvl_scores,
                                 max_output_boxes_per_class, iou_threshold,
                                 score_threshold, nms_pre, cfg.max_per_img)
    return det_results
