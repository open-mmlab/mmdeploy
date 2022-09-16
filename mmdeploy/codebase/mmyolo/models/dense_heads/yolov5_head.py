# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.codebase.mmyolo.models.layers import multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER
import copy


# @FUNCTION_REWRITER.register_rewriter(
#     func_name='mmyolo.models.dense_heads.yolov5_head.'
#     'YOLOv5Head.predict_by_feat')
# def yolov5_head__predict_by_feat(ctx,
#                                  self,
#                                  cls_scores: List[Tensor],
#                                  bbox_preds: List[Tensor],
#                                  objectnesses: Optional[List[Tensor]],
#                                  batch_img_metas: Optional[List[dict]] = None,
#                                  cfg: Optional[ConfigDict] = None,
#                                  rescale: bool = False,
#                                  with_nms: bool = True) -> List[InstanceData]:
#     """Rewrite `predict_by_feat` of `YOLOXHead` for default backend.
#
#     Rewrite this function to deploy model, transform network output for a
#     batch into bbox predictions.
#
#     Args:
#         ctx: Context that contains original meta information.
#         cls_scores (list[Tensor]): Classification scores for all
#             scale levels, each is a 4D-tensor, has shape
#             (batch_size, num_priors * num_classes, H, W).
#         bbox_preds (list[Tensor]): Box energies / deltas for all
#             scale levels, each is a 4D-tensor, has shape
#             (batch_size, num_priors * 4, H, W).
#         objectnesses (list[Tensor], Optional): Score factor for
#             all scale level, each is a 4D-tensor, has shape
#             (batch_size, 1, H, W).
#         batch_img_metas (list[dict], Optional): Batch image meta info.
#             Defaults to None.
#         cfg (ConfigDict, optional): Test / postprocessing
#             configuration, if None, test_cfg would be used.
#             Defaults to None.
#         rescale (bool): If True, return boxes in original image space.
#             Defaults to False.
#         with_nms (bool): If True, do nms before return boxes.
#             Defaults to True.
#
#     Returns:
#         tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
#             where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
#             size and the score between 0 and 1. The shape of the second
#             tensor in the tuple is (N, num_box), and each element
#             represents the class label of the corresponding box.
#     """
#     assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
#     cfg = self.test_cfg if cfg is None else cfg
#     batch_size = bbox_preds[0].shape[0]
#     featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
#     mlvl_priors = self.prior_generator.grid_priors(
#         featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
#     flatten_priors = torch.cat(mlvl_priors)
#
#     mlvl_strides = [
#         flatten_priors.new_full(
#             (featmap_size.numel() * self.num_base_priors, ), stride)
#         for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
#     ]
#
#     flatten_stride = torch.cat(mlvl_strides)
#
#     flatten_cls_scores = [
#         cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
#         for cls_score in cls_scores
#     ]
#     flatten_bbox_preds = [
#         bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
#         for bbox_pred in bbox_preds
#     ]
#     flatten_objectness = [
#         objectness.permute(0, 2, 3, 1).reshape(batch_size, -1)
#         for objectness in objectnesses
#     ]
#
#     cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
#     score_factor = torch.cat(flatten_objectness, dim=1).sigmoid()
#     flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
#
#     bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds,
#                                     flatten_stride)
#
#     # directly multiply score factor and feed to nms
#     scores = cls_scores * (score_factor.unsqueeze(-1))
#     max_scores, _ = torch.max(scores, 1)
#     mask = max_scores >= cfg.score_thr
#     scores = scores.where(mask, scores.new_zeros(1))
#     # add a pad for bboxes, or the results are wrong.
#     img_meta = batch_img_metas[0]
#     pad_param = img_meta['pad_param']
#     bboxes -= bboxes.new_tensor(
#         [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
#     if rescale:
#         scale_factor = img_meta['scale_factor']
#         bboxes /= bboxes.new_tensor(scale_factor).repeat((1, 2))
#     if not with_nms:
#         return bboxes, scores
#     deploy_cfg = ctx.cfg
#     post_params = get_post_processing_params(deploy_cfg)
#     max_output_boxes_per_class = post_params.max_output_boxes_per_class
#     iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
#     score_threshold = cfg.get('score_thr', post_params.score_threshold)
#     pre_top_k = post_params.pre_top_k
#     keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
#     keep_top_k = 20
#
#     return multiclass_nms(bboxes, scores, max_output_boxes_per_class,
#                           iou_threshold, score_threshold, pre_top_k,
#                           keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5Head.predict_by_feat')
def yolov5_head__predict_by_feat(ctx,
                                 self,
                                 cls_scores: List[Tensor],
                                 bbox_preds: List[Tensor],
                                 objectnesses: Optional[List[Tensor]],
                                 batch_img_metas: Optional[List[dict]] = None,
                                 cfg: Optional[ConfigDict] = None,
                                 rescale: bool = False,
                                 with_nms: bool = True) -> List[InstanceData]:
    assert len(cls_scores) == len(bbox_preds)
    if objectnesses is None:
        with_objectnesses = False
    else:
        with_objectnesses = True
        assert len(cls_scores) == len(objectnesses)

    cfg = self.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)

    multi_label = cfg.multi_label
    multi_label &= self.num_classes > 1
    cfg.multi_label = multi_label

    num_imgs = len(batch_img_metas)
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    # If the shape does not change, use the previous mlvl_priors
    if featmap_sizes != self.featmap_sizes:
        self.mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)
        self.featmap_sizes = featmap_sizes
    flatten_priors = torch.cat(self.mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size.numel() * self.num_base_priors, ), stride) for
        featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                              self.num_classes)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]

    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_decoded_bboxes = self.bbox_coder.decode(
        flatten_priors[None], flatten_bbox_preds, flatten_stride)

    if with_objectnesses:
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
    else:
        flatten_objectness = [None for _ in range(len(featmap_sizes))]

    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    score_factor = torch.cat(flatten_objectness, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds,
                                    flatten_stride)

    # directly multiply score factor and feed to nms
    scores = cls_scores * (score_factor.unsqueeze(-1))
    max_scores, _ = torch.max(scores, 1)
    mask = max_scores >= cfg.score_thr
    scores = scores.where(mask, scores.new_zeros(1))
    # add a pad for bboxes, or the results are wrong.
    img_meta = batch_img_metas[0]
    pad_param = img_meta['pad_param']
    bboxes -= bboxes.new_tensor(
        [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
    if rescale:
        scale_factor = img_meta['scale_factor']
        bboxes /= bboxes.new_tensor(scale_factor).repeat((1, 2))
    if not with_nms:
        return bboxes, scores
    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)
    keep_top_k = 20

    return multiclass_nms(bboxes, scores, max_output_boxes_per_class,
                          iou_threshold, score_threshold, pre_top_k,
                          keep_top_k)

    # results_list = []
    # for (bboxes, scores, objectness,
    #      img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
    #                       flatten_objectness, batch_img_metas):
    #     ori_shape = img_meta['ori_shape']
    #     scale_factor = img_meta['scale_factor']
    #     if 'pad_param' in img_meta:
    #         pad_param = img_meta['pad_param']
    #     else:
    #         pad_param = None
    #
    #     score_thr = cfg.get('score_thr', -1)
    #     # yolox_style does not require the following operations
    #     if objectness is not None and score_thr > 0 and not cfg.get(
    #             'yolox_style', False):
    #         conf_inds = objectness > score_thr
    #         bboxes = bboxes[conf_inds, :]
    #         scores = scores[conf_inds, :]
    #         objectness = objectness[conf_inds]
    #
    #     if objectness is not None:
    #         # conf = obj_conf * cls_conf
    #         scores *= objectness[:, None]
    #
    #     if scores.shape[0] == 0:
    #         empty_results = InstanceData()
    #         empty_results.bboxes = bboxes
    #         empty_results.scores = scores[:, 0]
    #         empty_results.labels = scores[:, 0].int()
    #         results_list.append(empty_results)
    #         continue
    #
    #     nms_pre = cfg.get('nms_pre', 100000)
    #     if cfg.multi_label is False:
    #         scores, labels = scores.max(1, keepdim=True)
    #         scores, _, keep_idxs, results = filter_scores_and_topk(
    #             scores,
    #             score_thr,
    #             nms_pre,
    #             results=dict(labels=labels[:, 0]))
    #         labels = results['labels']
    #     else:
    #         scores, labels, keep_idxs, _ = filter_scores_and_topk(
    #             scores, score_thr, nms_pre)
    #
    #     results = InstanceData(
    #         scores=scores, labels=labels, bboxes=bboxes[keep_idxs])
    #
    #     if rescale:
    #         if pad_param is not None:
    #             results.bboxes -= results.bboxes.new_tensor([
    #                 pad_param[2], pad_param[0], pad_param[2], pad_param[0]
    #             ])
    #         results.bboxes /= results.bboxes.new_tensor(
    #             scale_factor).repeat((1, 2))
    #
    #     if cfg.get('yolox_style', False):
    #         # do not need max_per_img
    #         cfg.max_per_img = len(results)
    #
    #     results = self._bbox_post_process(
    #         results=results,
    #         cfg=cfg,
    #         rescale=False,
    #         with_nms=with_nms,
    #         img_meta=img_meta)
    #     results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
    #     results.bboxes[:, 1::2].clamp_(0, ori_shape[0])
    #
    #     results_list.append(results)
    # return results_list
