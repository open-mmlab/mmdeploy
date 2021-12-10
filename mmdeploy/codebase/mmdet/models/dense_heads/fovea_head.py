# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet import get_post_processing_params, multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.FoveaHead.get_bboxes')
def fovea_head__get_bboxes(ctx,
                           self,
                           cls_scores,
                           bbox_preds,
                           score_factors=None,
                           img_metas=None,
                           cfg=None,
                           rescale=None,
                           **kwargs):
    """Rewrite `get_bboxes` of `FoveaHead` for default backend.

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
            Default: False.

    Returns:
        tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
    """
    assert len(cls_scores) == len(bbox_preds)
    cfg = self.test_cfg if cfg is None else cfg
    num_levels = len(cls_scores)
    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    points_list = self.get_points(
        featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device, flatten=True)
    cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
    img_shape = img_metas[0]['img_shape']
    batch_size = cls_scores[0].shape[0]

    det_bboxes = []
    det_scores = []
    for cls_score, bbox_pred, stride, base_len, (y, x) \
            in zip(cls_score_list, bbox_pred_list, self.strides,
                   self.base_edge_list, points_list):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        scores = cls_score.permute(0, 2, 3,
                                   1).reshape(batch_size, -1,
                                              self.cls_out_channels).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                          4).exp()
        x1 = (stride * x - base_len * bbox_pred[:, :, 0]). \
            clamp(min=0, max=img_shape[1] - 1)
        y1 = (stride * y - base_len * bbox_pred[:, :, 1]). \
            clamp(min=0, max=img_shape[0] - 1)
        x2 = (stride * x + base_len * bbox_pred[:, :, 2]). \
            clamp(min=0, max=img_shape[1] - 1)
        y2 = (stride * y + base_len * bbox_pred[:, :, 3]). \
            clamp(min=0, max=img_shape[0] - 1)
        bboxes = torch.stack([x1, y1, x2, y2], -1)
        det_bboxes.append(bboxes)
        det_scores.append(scores)
    det_bboxes = torch.cat(det_bboxes, dim=1)
    if rescale:
        scale_factor = img_metas['scale_factor']
        det_bboxes /= det_bboxes.new_tensor(scale_factor)
    det_scores = torch.cat(det_scores, dim=1)

    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    nms_pre = cfg.get('deploy_nms_pre', -1)
    det_results = multiclass_nms(det_bboxes, det_scores,
                                 max_output_boxes_per_class, iou_threshold,
                                 score_threshold, nms_pre, cfg.max_per_img)
    return det_results
