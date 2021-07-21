import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmdet.core import multiclass_nms
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter('mmdet.models.RPNHead.get_bboxes')
def get_bboxes_of_rpn_head(ctx,
                           self,
                           cls_scores,
                           bbox_preds,
                           img_metas,
                           with_nms=True,
                           cfg=None,
                           **kwargs):
    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)

    device = cls_scores[0].device
    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_anchors = self.anchor_generator.grid_anchors(
        featmap_sizes, device=device)

    mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(mlvl_anchors)

    cfg = self.test_cfg if cfg is None else cfg
    batch_size = mlvl_cls_scores[0].shape[0]
    pre_topk = cfg.get('nms_pre', -1)

    # loop over features, decode boxes
    mlvl_valid_bboxes = []
    mlvl_scores = []
    mlvl_valid_anchors = []
    for level_id, cls_score, bbox_pred, anchors in zip(
            range(num_levels), mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        cls_score = cls_score.permute(0, 2, 3, 1)
        if self.use_sigmoid_cls:
            cls_score = cls_score.reshape(batch_size, -1)
            scores = cls_score.sigmoid()
        else:
            cls_score = cls_score.reshape(batch_size, -1, 2)
            # We set FG labels to [0, num_class-1] and BG label to
            # num_class in RPN head since mmdet v2.5, which is unified to
            # be consistent with other head since mmdet v2.0. In mmdet v2.0
            # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
            scores = cls_score.softmax(-1)[..., 0]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        # use static anchor if input shape is static
        if not is_dynamic_flag:
            anchors = anchors.data

        anchors = anchors.expand_as(bbox_pred)

        enable_nms_pre = True
        backend = deploy_cfg['backend']
        # topk in tensorrt does not support shape<k
        # final level might meet the problem
        # TODO: support dynamic shape feature with TensorRT for topK op
        if backend == 'tensorrt':
            enable_nms_pre = (level_id != num_levels - 1)

        if pre_topk > 0 and enable_nms_pre:
            _, topk_inds = scores.topk(pre_topk)
            batch_inds = torch.arange(
                batch_size, device=device).view(-1, 1).expand_as(topk_inds)
            # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
            transformed_inds = scores.shape[1] * batch_inds + topk_inds
            scores = scores.reshape(-1, 1)[transformed_inds].reshape(
                batch_size, -1)
            bbox_pred = bbox_pred.reshape(-1, 4)[transformed_inds, :].reshape(
                batch_size, -1, 4)
            anchors = anchors.reshape(-1, 4)[transformed_inds, :].reshape(
                batch_size, -1, 4)
        mlvl_valid_bboxes.append(bbox_pred)
        mlvl_scores.append(scores)
        mlvl_valid_anchors.append(anchors)

    batch_mlvl_bboxes = torch.cat(mlvl_valid_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1).unsqueeze(2)
    batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
    batch_mlvl_bboxes = self.bbox_coder.decode(
        batch_mlvl_anchors,
        batch_mlvl_bboxes,
        max_shape=img_metas['img_shape'])
    # ignore background class
    if not self.use_sigmoid_cls:
        batch_mlvl_scores = batch_mlvl_scores[..., :self.num_classes]
    if not with_nms:
        return batch_mlvl_bboxes, batch_mlvl_scores

    post_params = deploy_cfg.post_processing
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    # only one class in rpn
    max_output_boxes_per_class = keep_top_k
    return multiclass_nms(
        batch_mlvl_bboxes,
        batch_mlvl_scores,
        max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)
