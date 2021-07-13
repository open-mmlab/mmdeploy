import torch

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmdet.core import distance2bbox, multiclass_nms
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.FCOSHead.get_bboxes')
def get_bboxes_of_fcos_head(rewriter,
                            self,
                            cls_scores,
                            bbox_preds,
                            centernesses,
                            img_shape,
                            with_nms=True,
                            cfg=None,
                            **kwargs):
    assert len(cls_scores) == len(bbox_preds)
    deploy_cfg = rewriter.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    num_levels = len(cls_scores)

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                  bbox_preds[0].device)

    cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
    centerness_pred_list = [
        centernesses[i].detach() for i in range(num_levels)
    ]

    cfg = self.test_cfg if cfg is None else cfg
    assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
    device = cls_scores[0].device
    batch_size = cls_scores[0].shape[0]
    pre_topk = cfg.get('nms_pre', -1)

    # loop over features, decode boxes
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_centerness = []
    for level_id, cls_score, bbox_pred, centerness, points in zip(
            range(num_levels), cls_score_list, bbox_pred_list,
            centerness_pred_list, mlvl_points):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        scores = cls_score.permute(0, 2, 3,
                                   1).reshape(batch_size, -1,
                                              self.cls_out_channels).sigmoid()
        centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size,
                                                            -1).sigmoid()

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        # use static anchor if input shape is static
        if not is_dynamic_flag:
            points = points.data

        points = points.expand(batch_size, -1, 2)

        enable_nms_pre = True
        backend = deploy_cfg['backend']
        # topk in tensorrt does not support shape<k
        # final level might meet the problem
        if backend == 'tensorrt':
            enable_nms_pre = (level_id != num_levels - 1)

        if pre_topk > 0 and enable_nms_pre:
            max_scores, _ = (scores * centerness[..., None]).max(-1)
            _, topk_inds = max_scores.topk(pre_topk)
            batch_inds = torch.arange(batch_size).view(
                -1, 1).expand_as(topk_inds).long().to(device)
            # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501

            transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
            points = points.reshape(-1, 2)[transformed_inds, :].reshape(
                batch_size, -1, 2)
            bbox_pred = bbox_pred.reshape(-1, 4)[transformed_inds, :].reshape(
                batch_size, -1, 4)
            scores = scores.reshape(
                -1, self.num_classes)[transformed_inds, :].reshape(
                    batch_size, -1, self.num_classes)
            centerness = centerness.reshape(-1, 1)[transformed_inds].reshape(
                batch_size, -1)

        bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_centerness.append(centerness)

    batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

    if not with_nms:
        return batch_mlvl_bboxes, batch_mlvl_scores, batch_mlvl_centerness

    batch_mlvl_scores = batch_mlvl_scores * (
        batch_mlvl_centerness.unsqueeze(2))
    max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class', 200)
    iou_threshold = cfg.nms.get('iou_threshold', 0.5)
    score_threshold = cfg.score_thr
    nms_pre = cfg.get('deploy_nms_pre', -1)
    return multiclass_nms(batch_mlvl_bboxes, batch_mlvl_scores,
                          max_output_boxes_per_class, iou_threshold,
                          score_threshold, nms_pre, cfg.max_per_img)


@FUNCTION_REWRITER.register_rewriter('mmdet.models.FCOSHead.forward')
@mark('rpn_forward', outputs=['cls_score', 'bbox_pred', 'centerness'])
def forward_of_fcos_head(rewriter, *args):
    return rewriter.origin_func(*args)
