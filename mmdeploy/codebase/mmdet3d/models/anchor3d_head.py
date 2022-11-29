# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet3d.core.bbox.structures import limit_period

from mmdeploy.codebase.mmdet3d.core.post_processing import box3d_multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.dense_heads.anchor3d_head.Anchor3DHead.'
    'get_bboxes')
def anchor3dhead__get_bboxes(ctx,
                             self,
                             cls_scores,
                             bbox_preds,
                             dir_cls_preds,
                             input_metas,
                             cfg=None,
                             rescale=False):
    """Rewrite `get_bboxes` of `Anchor3DHead` for default backend.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (FoveaHead): The instance of the class FoveaHead.
        cls_scores (list[Tensor]): Box scores for each scale level
            with shape (N, num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (N, num_anchors * 7, H, W).
        dir_cls_preds (list[Tensor]): Direction predicts for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W).
        input_metas (list[dict]):  Meta information of the image, e.g.,
            image size, scaling factor, etc.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used. Default: None.
        rescale (bool): If True, return boxes in original image space.
            Default: False.

    Returns:
        tuple[Tensor, Tensor]: tuple[Tensor, Tensor]: (bboxes, scores, labels),
            `bboxes` of shape [N, num_det, 7] ,`scores` of shape
            [N, num_det] and `labels` of shape [N, num_det].
    """
    assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds)
    num_levels = len(cls_scores)
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    device = cls_scores[0].device
    mlvl_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device=device)
    mlvl_anchors = [
        anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
    ]

    cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    dir_cls_preds = [dir_cls_preds[i].detach() for i in range(num_levels)]

    cfg = self.test_cfg if cfg is None else cfg
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_dir_scores = []
    for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
            cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 1).reshape(1, -1, 2)
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(1, -1, self.num_classes)
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(1, -1, self.box_code_size)

        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[1] > nms_pre:
            if self.use_sigmoid_cls:
                max_scores, _ = scores.max(dim=2)
            else:
                max_scores, _ = scores[..., :-1].max(dim=2)
            max_scores = max_scores[0]
            _, topk_inds = max_scores.topk(nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[:, topk_inds, :]
            scores = scores[:, topk_inds, :]
            dir_cls_score = dir_cls_score[:, topk_inds]

        bboxes = self.bbox_coder.decode(anchors, bbox_pred)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_dir_scores.append(dir_cls_score)

    mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    mlvl_bboxes_for_nms = mlvl_bboxes[..., [0, 1, 3, 4, 6]].clone()
    mlvl_scores = torch.cat(mlvl_scores, dim=1)
    mlvl_dir_scores = torch.cat(mlvl_dir_scores, dim=1)
    if mlvl_bboxes.shape[0] > 0:
        dir_rot = limit_period(mlvl_bboxes[..., 6] - self.dir_offset,
                               self.dir_limit_offset, np.pi)
        mlvl_bboxes[..., 6] = (
            dir_rot + self.dir_offset +
            np.pi * mlvl_dir_scores.to(mlvl_bboxes.dtype))
    return box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores,
                                cfg.score_thr, cfg.nms_thr, cfg.max_num)
