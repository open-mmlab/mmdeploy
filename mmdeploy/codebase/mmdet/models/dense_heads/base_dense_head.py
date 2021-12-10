import torch

from mmdeploy.codebase.mmdet import (get_post_processing_params,
                                     multiclass_nms, pad_with_value)
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend, get_backend, is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.base_dense_head.'
    'BaseDenseHead.get_bboxes')
def base_dense_head__get_bbox(ctx,
                              self,
                              cls_scores,
                              bbox_preds,
                              score_factors=None,
                              img_metas=None,
                              cfg=None,
                              rescale=False,
                              with_nms=True,
                              **kwargs):
    """Rewrite `get_bboxes` of `BaseDenseHead` for default backend.

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
    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    backend = get_backend(deploy_cfg)
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)

    mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

    assert len(
        img_metas
    ) == 1, 'Only support one input image while in exporting to ONNX'
    img_shape = img_metas[0]['img_shape']

    cfg = self.test_cfg
    assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
    device = cls_scores[0].device
    batch_size = cls_scores[0].shape[0]
    # convert to tensor to keep tracing
    nms_pre_tensor = torch.tensor(
        cfg.get('nms_pre', -1), device=device, dtype=torch.long)
    # e.g. Retina, FreeAnchor, etc.
    if score_factors is None:
        with_score_factors = False
        mlvl_score_factor = [None for _ in range(num_levels)]
    else:
        # e.g. FCOS, PAA, ATSS, etc.
        with_score_factors = True
        mlvl_score_factor = [
            score_factors[i].detach() for i in range(num_levels)
        ]
        mlvl_score_factors = []

    mlvl_batch_bboxes = []
    mlvl_scores = []

    for cls_score, bbox_pred, score_factors, priors in zip(
            mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor, mlvl_priors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

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
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        if not is_dynamic_flag:
            priors = priors.data
        priors = priors.expand(batch_size, -1, priors.size(-1))
        # Get top-k predictions
        from mmdet.core.export import get_k_for_topk
        size = torch.tensor(bbox_pred.shape[1], device=device)
        nms_pre = get_k_for_topk(nms_pre_tensor, size)

        if nms_pre > 0:
            if with_score_factors:
                nms_pre_score = (nms_pre_score * score_factors[..., None])
            else:
                nms_pre_score = nms_pre_score
            if backend == Backend.TENSORRT:
                priors = pad_with_value(priors, 1, nms_pre_tensor)
                bbox_pred = pad_with_value(bbox_pred, 1, nms_pre_tensor)
                nms_pre_score = pad_with_value(nms_pre_score, 1,
                                               nms_pre_tensor, 0.)
            # Get maximum scores for foreground classes.
            if self.use_sigmoid_cls:
                max_scores, _ = nms_pre_score.max(-1)
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                max_scores, _ = nms_pre_score[..., :-1].max(-1)
            _, topk_inds = max_scores.topk(nms_pre)

            batch_inds = torch.arange(
                batch_size,
                device=bbox_pred.device).view(-1,
                                              1).expand_as(topk_inds).long()
            priors = priors[batch_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]

            if with_score_factors:
                score_factors = score_factors.unsqueeze(2)[batch_inds,
                                                           topk_inds, :]

        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        mlvl_batch_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        if with_score_factors:
            mlvl_score_factors.append(score_factors)

    batch_bboxes = torch.cat(mlvl_batch_bboxes, dim=1)
    batch_scores = torch.cat(mlvl_scores, dim=1)
    if with_score_factors:
        batch_score_factors = torch.cat(mlvl_score_factors, dim=1)

    # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment

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
