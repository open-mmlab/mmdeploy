# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet3d.core.bbox import points_img2cam
from mmdeploy.codebase.mmdet3d.core.post_processing import box3d_multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.dense_heads.fcos_mono3d_head.FCOSMono3DHead.get_bboxes')
def fcosmono3dhead__get_bboxes(
    ctx,
    self,
    cls_scores,
    bbox_preds,
    dir_cls_preds,
    attr_preds,
    centernesses,
    cam2img,
    cam2img_inverse,
    img_meta,
    cfg=None,
    rescale=False,
):
    """Rewrite `get_bboxes` of `FCOSMono3DHead` for default backend.
    Rewrite this function to deploy model, transform network output for a
    batch into 3d bbox predictions.
    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 9, H, W).
        dir_cls_preds (list[Tensor], Optional): Direction prediction for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W).
        attr_preds (list[Tensor], Optional): Attribute scores for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, num_priors * 1, H, W).
        img_metas (list[dict], Optional): Image meta info. Default None.
        cfg (mmcv.Config, Optional): Test / postprocessing configuration,
            if None, test_cfg would be used.  Default None.
        rescale (bool): If True, return boxes in original image space.
            Default False.
    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: batch_bboxes,
        batch_scores, batch_labels, batch_dir_pred, batch_attr.
    """
    assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == len(
        centernesses) == len(attr_preds)
    num_levels = len(cls_scores)
    batch_size = 1

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                  bbox_preds[0].device)

    cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    dir_cls_preds = [dir_cls_preds[i].detach() for i in range(num_levels)]
    attr_preds = [attr_preds[i].detach() for i in range(num_levels)]
    centernesses = [centernesses[i].detach() for i in range(num_levels)]

    input_meta = img_meta[0]
    scale_factor = input_meta['scale_factor']
    cfg = self.test_cfg if cfg is None else cfg

    mlvl_centers2d = []
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_dir_scores = []
    mlvl_attr_scores = []
    mlvl_centerness = []
    for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
        points in zip(cls_scores, bbox_preds, dir_cls_preds,
                      attr_preds, centernesses, mlvl_points):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        scores = cls_score.permute(0, 2, 3,
                                   1).reshape(batch_size, -1,
                                              self.cls_out_channels).sigmoid()
        dir_cls_pred = dir_cls_pred.permute(0, 2, 3,
                                            1).reshape(batch_size, -1, 2)
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
        attr_pred = attr_pred.permute(0, 2, 3,
                                      1).reshape(batch_size, -1,
                                                 self.num_attrs)
        attr_score = torch.max(attr_pred, dim=-1)[1]
        centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size,
                                                            -1).sigmoid()

        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(batch_size, -1,
                                                 sum(self.group_reg_dims))
        bbox_pred = bbox_pred[..., :self.bbox_code_size]
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[1] > nms_pre:
            max_scores, _ = (scores * centerness[..., None]).max(dim=2)
            max_scores = max_scores[0]
            _, topk_inds = max_scores.topk(nms_pre)
            points = points[topk_inds, :]
            bbox_pred = bbox_pred[:, topk_inds, :]
            scores = scores[:, topk_inds, :]
            dir_cls_pred = dir_cls_pred[:, topk_inds, :]
            centerness = centerness[:, topk_inds]
            dir_cls_score = dir_cls_score[:, topk_inds]
            attr_score = attr_score[:, topk_inds]
        # change the offset to actual center predictions
        bbox_pred[..., :2] = points.unsqueeze(0) - bbox_pred[..., :2]
        if rescale:
            bbox_pred[..., :2] /= bbox_pred[..., :2].new_tensor(scale_factor)
        pred_center2d = bbox_pred[..., :3].clone()
        bbox_pred[..., :3] = points_img2cam(bbox_pred[..., :3],
                                            cam2img_inverse)
        mlvl_centers2d.append(pred_center2d)
        mlvl_bboxes.append(bbox_pred)
        mlvl_scores.append(scores)
        mlvl_dir_scores.append(dir_cls_score)
        mlvl_attr_scores.append(attr_score)
        mlvl_centerness.append(centerness)
    mlvl_centers2d = torch.cat(mlvl_centers2d, dim=1)
    mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    mlvl_dir_scores = torch.cat(mlvl_dir_scores, dim=1)

    # change local yaw to global yaw for 3D nms
    mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d,
                                             mlvl_dir_scores, self.dir_offset,
                                             cam2img)

    mlvl_bboxes_for_nms = mlvl_bboxes[..., [0, 2, 3, 5, 6]].clone()
    mlvl_bboxes_for_nms[..., -1] = -mlvl_bboxes_for_nms[..., -1]

    mlvl_scores = torch.cat(mlvl_scores, dim=1)
    mlvl_attr_scores = torch.cat(mlvl_attr_scores, dim=1)
    mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
    # no scale_factors in box3d_multiclass_nms
    # Then we multiply it from outside
    mlvl_nms_scores = mlvl_scores * mlvl_centerness[..., None]
    return box3d_multiclass_nms(
        mlvl_bboxes,
        mlvl_bboxes_for_nms,
        mlvl_nms_scores,
        cfg.score_thr,
        cfg.nms_thr,
        cfg.max_per_img,
        mlvl_dir_scores,
        mlvl_attr_scores,
    )
