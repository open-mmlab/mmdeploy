import torch

import mmdeploy
from mmdeploy.utils import FUNCTION_REWRITERS, is_dynamic_shape


@FUNCTION_REWRITERS.register_rewriter(
    func_name='mmdet.models.AnchorHead.get_bboxes')
@FUNCTION_REWRITERS.register_rewriter(
    func_name='mmdet.models.RetinaHead.get_bboxes')
def anchor_head_get_bboxes(rewriter,
                           self,
                           cls_scores,
                           bbox_preds,
                           img_shape,
                           with_nms=True,
                           cfg=None,
                           **kwargs):
    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = rewriter.cfg
    num_levels = len(cls_scores)

    device = cls_scores[0].device
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device=device)

    mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

    cfg = self.test_cfg if cfg is None else cfg
    assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)
    batch_size = mlvl_cls_scores[0].shape[0]
    nms_pre = cfg.get('nms_pre', -1)

    # loop over features, decode boxes
    mlvl_bboxes = []
    mlvl_scores = []
    for level_id, cls_score, bbox_pred, anchors in zip(
            range(num_levels), mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(batch_size, -1,
                                                 self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        # use static anchor if input shape is static
        if not is_dynamic_shape(deploy_cfg):
            anchors = anchors.data

        anchors = anchors.expand_as(bbox_pred)

        enable_nms_pre = True
        backend = deploy_cfg['backend']
        # topk in tensorrt does not support shape<k
        # final level might meet the problem
        if backend == 'tensorrt':
            enable_nms_pre = (level_id != num_levels - 1)

        if nms_pre > 0 and enable_nms_pre:
            # Get maximum scores for foreground classes.
            if self.use_sigmoid_cls:
                max_scores, _ = scores.max(-1)
            else:

                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                max_scores, _ = scores[..., :-1].max(-1)
            _, topk_inds = max_scores.topk(nms_pre)
            batch_inds = torch.arange(batch_size).view(-1,
                                                       1).expand_as(topk_inds)
            anchors = anchors[batch_inds, topk_inds, :]
            bbox_pred = bbox_pred[batch_inds, topk_inds, :]
            scores = scores[batch_inds, topk_inds, :]

        if not is_dynamic_shape(deploy_cfg):
            img_shape = [int(val) for val in img_shape]

        bboxes = self.bbox_coder.decode(
            anchors, bbox_pred, max_shape=img_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)

    batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

    # ignore background class
    if not self.use_sigmoid_cls:
        batch_mlvl_scores = batch_mlvl_scores[..., :self.num_classes]
    if not with_nms:
        return batch_mlvl_bboxes, batch_mlvl_scores

    max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class', 200)
    iou_threshold = cfg.nms.get('iou_threshold', 0.5)
    score_threshold = cfg.score_thr
    nms_pre = cfg.get('deploy_nms_pre', -1)
    return mmdeploy.mmdet.core.export.add_dummy_nms_for_onnx(
        batch_mlvl_bboxes,
        batch_mlvl_scores,
        max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=nms_pre,
        after_top_k=cfg.max_per_img)
