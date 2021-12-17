# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet import get_post_processing_params, multiclass_nms
from mmdeploy.codebase.mmdet.core.ops.detection_output import \
    ncnn_detection_output_forward
from mmdeploy.codebase.mmdet.core.ops.prior_box import ncnn_prior_box_forward
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.AnchorHead.get_bboxes', backend='ncnn')
def anchor_head__get_bboxes__ncnn(ctx,
                                  self,
                                  cls_scores,
                                  bbox_preds,
                                  img_metas,
                                  with_nms=True,
                                  cfg=None,
                                  **kwargs):
    """Rewrite `get_bboxes` of AnchorHead for NCNN backend.

    Shape node and batch inference is not supported by ncnn. This function
    transform dynamic shape to constant shape and remove batch inference.

    Args:
        ctx (ContextCaller): The context with additional information.
        cls_scores (list[Tensor]): Box scores for each level in the
            feature pyramid, has shape
            (N, num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each
            level in the feature pyramid, has shape
            (N, num_anchors * 4, H, W).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used.
            Default: None.


    Returns:
        If isinstance(self, SSDHead) == True:
            Tensor: outputs, shape is [N, num_det, 5].
        If isinstance(self, SSDHead) == False:
            If with_nms == True:
                tuple[Tensor, Tensor]: (dets, labels),
                `dets` of shape [N, num_det, 5] and `labels` of shape
                [N, num_det].
            Else:
                tuple[Tensor, Tensor]: batch_mlvl_bboxes, batch_mlvl_scores
    """
    from mmdet.models.dense_heads import SSDHead

    # now the ncnn PriorBox and DetectionOutput adaption is only used for
    # SSDHead.
    # TODO: Adapt all of the AnchorHead instances for ncnn PriorBox and
    # DetectionOutput. Then, the determine statement will be removed, and
    # the code will be unified.
    if not isinstance(self, SSDHead):
        assert len(cls_scores) == len(bbox_preds)
        deploy_cfg = ctx.cfg
        assert not is_dynamic_shape(deploy_cfg)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) \
            == len(mlvl_anchors)
        batch_size = 1
        pre_topk = cfg.get('nms_pre', -1)

        # loop over features, decode boxes
        mlvl_valid_bboxes = []
        mlvl_valid_anchors = []
        mlvl_scores = []
        for level_id, cls_score, bbox_pred, anchors in zip(
                range(num_levels), mlvl_cls_scores, mlvl_bbox_preds,
                mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).\
                reshape(batch_size, -1, 4)

            # use static anchor if input shape is static
            anchors = anchors.expand_as(bbox_pred).data

            if pre_topk > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)
                _, topk_inds = max_scores.topk(pre_topk)

                topk_inds = topk_inds.view(-1)
                anchors = anchors[:, topk_inds, :]
                bbox_pred = bbox_pred[:, topk_inds, :]
                scores = scores[:, topk_inds, :]

            mlvl_valid_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_valid_anchors.append(anchors)

        batch_mlvl_valid_bboxes = torch.cat(mlvl_valid_bboxes, dim=1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_anchors = torch.cat(mlvl_valid_anchors, dim=1)
        batch_mlvl_bboxes = self.bbox_coder.decode(
            batch_mlvl_anchors,
            batch_mlvl_valid_bboxes,
            max_shape=img_metas[0]['img_shape'])

        # ignore background class
        if not self.use_sigmoid_cls:
            batch_mlvl_scores = batch_mlvl_scores[..., :self.num_classes]
        if not with_nms:
            return batch_mlvl_bboxes, batch_mlvl_scores

        post_params = get_post_processing_params(deploy_cfg)
        max_output_boxes_per_class = post_params.max_output_boxes_per_class
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        score_threshold = cfg.get('score_thr', post_params.score_threshold)
        pre_top_k = post_params.pre_top_k
        keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
        return multiclass_nms(
            batch_mlvl_bboxes,
            batch_mlvl_scores,
            max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)
    else:
        assert len(cls_scores) == len(bbox_preds)
        deploy_cfg = ctx.cfg
        num_levels = len(cls_scores)
        aspect_ratio = [
            ratio[ratio > 1].detach().cpu().numpy()
            for ratio in self.anchor_generator.ratios
        ]
        min_sizes = self.anchor_generator.base_sizes
        max_sizes = min_sizes[1:] + \
            img_metas[0]['img_shape'][0:1].tolist()
        img_height = img_metas[0]['img_shape'][0].item()
        img_width = img_metas[0]['img_shape'][1].item()

        # if no reshape, concat will be error in ncnn.
        mlvl_anchors = [
            ncnn_prior_box_forward(cls_scores[i], aspect_ratio[i], img_height,
                                   img_width, max_sizes[i:i + 1],
                                   min_sizes[i:i + 1]).reshape(1, 2, -1)
            for i in range(num_levels)
        ]

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) \
            == len(mlvl_anchors)
        batch_size = 1

        mlvl_valid_bboxes = []
        mlvl_scores = []
        for level_id, cls_score, bbox_pred in zip(
                range(num_levels), mlvl_cls_scores, mlvl_bbox_preds):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1). \
                reshape(batch_size, -1, 4)

            mlvl_valid_bboxes.append(bbox_pred)
            mlvl_scores.append(cls_score)

        # NCNN DetectionOutput layer uses background class at 0 position, but
        # in mmdetection, background class is at self.num_classes position.
        # We should adapt for ncnn.
        batch_mlvl_valid_bboxes = torch.cat(mlvl_valid_bboxes, dim=1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        if self.use_sigmoid_cls:
            batch_mlvl_scores = batch_mlvl_scores.sigmoid()
        else:
            batch_mlvl_scores = batch_mlvl_scores.softmax(-1)
        batch_mlvl_anchors = torch.cat(mlvl_anchors, dim=2)
        batch_mlvl_scores = torch.cat([
            batch_mlvl_scores[:, :, self.num_classes:],
            batch_mlvl_scores[:, :, 0:self.num_classes]
        ],
                                      dim=2)
        batch_mlvl_valid_bboxes = batch_mlvl_valid_bboxes. \
            reshape(batch_size, 1, -1)
        batch_mlvl_scores = batch_mlvl_scores.reshape(batch_size, 1, -1)
        batch_mlvl_anchors = batch_mlvl_anchors.reshape(batch_size, 2, -1)

        post_params = get_post_processing_params(deploy_cfg)
        iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
        score_threshold = cfg.get('score_thr', post_params.score_threshold)
        pre_top_k = post_params.pre_top_k
        keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

        output__ncnn = ncnn_detection_output_forward(
            batch_mlvl_valid_bboxes, batch_mlvl_scores, batch_mlvl_anchors,
            score_threshold, iou_threshold, pre_top_k, keep_top_k,
            self.num_classes + 1)

        return output__ncnn
