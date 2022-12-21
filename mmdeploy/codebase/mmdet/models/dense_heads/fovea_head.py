# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops import multiclass_nms


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.fovea_head.FoveaHead.predict_by_feat')
def fovea_head__predict_by_feat(self,
                                cls_scores: List[Tensor],
                                bbox_preds: List[Tensor],
                                score_factors: Optional[List[Tensor]] = None,
                                batch_img_metas: Optional[List[dict]] = None,
                                cfg: Optional[ConfigDict] = None,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
    """Rewrite `predict_by_feat` of `FoveaHead` for default backend.

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
        batch_img_metas (list[dict]):  Meta information of the image, e.g.,
            image size, scaling factor, etc.
        cfg (mmengine.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.
        rescale (bool): If True, return boxes in original image space.
            Default: False.

    Returns:
        tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (dets, labels),
            `dets` of shape [N, num_det, 5] and `labels` of shape
            [N, num_det].
    """
    ctx = FUNCTION_REWRITER.get_context()
    assert len(cls_scores) == len(bbox_preds)
    cfg = self.test_cfg if cfg is None else cfg
    num_levels = len(cls_scores)
    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
    cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
    img_shape = batch_img_metas[0]['img_shape']
    batch_size = cls_scores[0].shape[0]

    det_bboxes = []
    det_scores = []
    for cls_score, bbox_pred, base_len, point \
            in zip(cls_score_list, bbox_pred_list,
                   self.base_edge_list, mlvl_priors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        x = point[:, 0]
        y = point[:, 1]
        scores = cls_score.permute(0, 2, 3,
                                   1).reshape(batch_size, -1,
                                              self.cls_out_channels).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                          4).exp()
        x1 = (x - base_len * bbox_pred[:, :, 0]). \
            clamp(min=0, max=img_shape[1] - 1)
        y1 = (y - base_len * bbox_pred[:, :, 1]). \
            clamp(min=0, max=img_shape[0] - 1)
        x2 = (x + base_len * bbox_pred[:, :, 2]). \
            clamp(min=0, max=img_shape[1] - 1)
        y2 = (y + base_len * bbox_pred[:, :, 3]). \
            clamp(min=0, max=img_shape[0] - 1)
        bboxes = torch.stack([x1, y1, x2, y2], -1)
        det_bboxes.append(bboxes)
        det_scores.append(scores)
    det_bboxes = torch.cat(det_bboxes, dim=1)
    if rescale:
        scale_factor = batch_img_metas['scale_factor']
        det_bboxes /= det_bboxes.new_tensor(scale_factor)
    det_scores = torch.cat(det_scores, dim=1)

    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    nms_pre = cfg.get('deploy_nms_pre', -1)
    nms_type = cfg.nms.get('type')
    return multiclass_nms(
        det_bboxes,
        det_scores,
        max_output_boxes_per_class,
        nms_type=nms_type,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=nms_pre,
        keep_top_k=cfg.max_per_img)
