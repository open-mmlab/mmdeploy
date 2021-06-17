import torch
import torch.nn as nn

from mmdeploy.utils import MODULE_REWRITERS
from mmdeploy.mmdet.core.export import add_dummy_nms_for_onnx

@MODULE_REWRITERS.register_rewrite_module(module_type='mmdet.models.AnchorHead')
@MODULE_REWRITERS.register_rewrite_module(module_type='mmdet.models.RetinaHead')
class AnchorHead(nn.Module):
    def __init__(self, module, cfg, **kwargs):
        super(AnchorHead, self).__init__()
        self.module = module
        self.anchor_generator = self.module.anchor_generator
        self.bbox_coder = self.module.bbox_coder

        self.test_cfg = module.test_cfg
        self.num_classes = module.num_classes
        self.use_sigmoid_cls = module.use_sigmoid_cls
        self.cls_out_channels = module.cls_out_channels

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_shape,
                   with_nms=True,
                   cfg=None,
                   **kwargs):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)
                # _, topk_inds = torch.topk(max_scores, nms_pre)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

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
        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        # ignore background class
        if not self.use_sigmoid_cls:
            num_classes = batch_mlvl_scores.shape[2] - 1
            batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
        max_output_boxes_per_class = cfg.nms.get(
            'max_output_boxes_per_class', 200)
        iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        score_threshold = cfg.score_thr
        nms_pre = cfg.get('deploy_nms_pre', -1)
        return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                      max_output_boxes_per_class,
                                      iou_threshold, score_threshold,
                                      nms_pre, cfg.max_per_img)


