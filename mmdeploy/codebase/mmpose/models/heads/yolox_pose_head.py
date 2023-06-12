# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import copy
from mmdeploy.codebase.mmdet import get_post_processing_params

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.models.utils import filter_scores_and_topk
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='models.yolox_pose_head.'
              'YOLOXPoseHead.predict')
def predict(self,
            x: Tuple[Tensor],
            batch_data_samples,
            rescale: bool = False):
    batch_img_metas = [
        data_samples.metainfo for data_samples in batch_data_samples
    ]

    outs = self(x)

    predictions = self.predict_by_feat(
        *outs, batch_img_metas=batch_img_metas, rescale=rescale)
    return predictions


@FUNCTION_REWRITER.register_rewriter(
    func_name='models.yolox_pose_head.'
              'YOLOXPoseHead.predict_by_feat')
def yolox_pose_head__predict_by_feat(self,
                                     cls_scores: List[Tensor],
                                     bbox_preds: List[Tensor],
                                     objectnesses: Optional[List[Tensor]] = None,
                                     kpt_preds: Optional[List[Tensor]] = None,
                                     vis_preds: Optional[List[Tensor]] = None,
                                     batch_img_metas: Optional[List[dict]] = None,
                                     cfg: Optional[ConfigDict] = None,
                                     rescale: bool = True,
                                     with_nms: bool = True) -> List[InstanceData]:
    """Transform a batch of output features extracted by the head into bbox
    and keypoint results.

    In addition to the base class method, keypoint predictions are also
    calculated in this method.
    """
    ctx = FUNCTION_REWRITER.get_context()
    deploy_cfg = ctx.cfg
    dtype = cls_scores[0].dtype
    device = cls_scores[0].device
    bbox_decoder = self.bbox_coder.decode

    assert len(cls_scores) == len(bbox_preds)
    cfg = self.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)

    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    self.mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=dtype, device=device)

    flatten_priors = torch.cat(self.mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size[0] * featmap_size[1] * self.num_base_priors,),
            stride)
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
        for cls_score in cls_scores
    ]
    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    if objectnesses is not None:
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

    scores = cls_scores

    bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds,
                          flatten_stride)

    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)

    priors = torch.cat(self.mlvl_priors)
    strides = [
        priors.new_full((featmap_size.numel() * self.num_base_priors,),
                        stride) for featmap_size, stride in zip(
            featmap_sizes, self.featmap_strides)
    ]
    strides = torch.cat(strides)
    kpt_preds = torch.cat([
        kpt_pred.permute(0, 2, 3, 1).reshape(
            num_imgs, -1, self.num_keypoints * 2) for kpt_pred in kpt_preds
    ],
        dim=1)
    flatten_decoded_kpts = self.decode_pose(priors, kpt_preds, strides)

    vis_preds = torch.cat([
        vis_pred.permute(0, 2, 3, 1).reshape(
            num_imgs, -1, self.num_keypoints) for vis_pred in vis_preds
    ],
        dim=1).sigmoid()

    nms_pre = cfg.get('nms_pre', 100000)
    score_thr = cfg.get('score_thr', -1)
    result_list = []
    for batch_idx in range(len(batch_img_metas)):
        if cfg.multi_label is False:
            pred_scores, pred_labels = scores[batch_idx].max(1, keepdim=True)
            pred_score, _, keep_idxs, results = filter_scores_and_topk(
                scores[batch_idx],
                score_thr,
                nms_pre,
                results=dict(labels=pred_labels[:, 0]))
            pred_label = results['labels']
        else:
            pred_score, pred_label, keep_idxs, _ = filter_scores_and_topk(scores[batch_idx], score_thr, nms_pre)

        pred_bbox = bboxes[batch_idx][keep_idxs]
        kpts = flatten_decoded_kpts[batch_idx, keep_idxs]
        kpts_vis = vis_preds[batch_idx, keep_idxs]

        from mmdeploy.mmcv.ops.nms import ONNXNMSop
        from packaging import version

        def yolox_pose_head_nms(boxes: Tensor,
                                scores: Tensor,
                                max_output_boxes_per_class: int = 1000,
                                iou_threshold: float = 0.5,
                                score_threshold: float = 0.05,
                                pre_top_k: int = -1,
                                keep_top_k: int = -1,
                                output_index: bool = True):
            if version.parse(torch.__version__) < version.parse('1.13.0'):
                max_output_boxes_per_class = torch.LongTensor(
                    [max_output_boxes_per_class])
            iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
            score_threshold = torch.tensor([score_threshold], dtype=torch.float32)

            # pre topk
            if pre_top_k > 0:
                max_scores, _ = scores.max(-1)
                _, topk_inds = max_scores.squeeze(0).topk(pre_top_k)
                boxes = boxes[:, topk_inds, :]
                scores = scores[:, topk_inds, :]

            scores = scores.permute(0, 2, 1)
            selected_indices = ONNXNMSop.apply(boxes, scores,
                                               max_output_boxes_per_class,
                                               iou_threshold, score_threshold)

            cls_inds = selected_indices[:, 1]
            box_inds = selected_indices[:, 2]

            scores = scores[:, cls_inds, box_inds].unsqueeze(2)
            boxes = boxes[:, box_inds, ...]
            dets = torch.cat([boxes, scores], dim=2)
            labels = cls_inds.unsqueeze(0)

            # pad
            dets = torch.cat((dets, dets.new_zeros((1, 1, 5))), 1)
            labels = torch.cat((labels, labels.new_zeros((1, 1))), 1)

            # topk or sort
            is_use_topk = keep_top_k > 0 and \
                          (torch.onnx.is_in_onnx_export() or keep_top_k < dets.shape[1])
            if is_use_topk:
                _, topk_inds = dets[:, :, -1].topk(keep_top_k, dim=1)
            else:
                _, topk_inds = dets[:, :, -1].sort(dim=1, descending=True)
            topk_inds = topk_inds.squeeze(0)
            dets = dets[:, topk_inds, ...]
            labels = labels[:, topk_inds, ...]

            if output_index:
                return dets, labels, box_inds
            else:
                return dets, labels

        nms_result = yolox_pose_head_nms(pred_bbox.unsqueeze(0), pred_score.reshape(-1, 1).unsqueeze(0),
                                         max_output_boxes_per_class, iou_threshold, score_threshold)
        keep_indices_nms = [nms_result[2]]

        img_meta = batch_img_metas[batch_idx]
        if rescale:
            pad_param = img_meta.get('pad_param', None)
            scale_factor = img_meta['scale_factor']
            if pad_param is not None:
                kpts -= kpts.new_tensor([pad_param[2], pad_param[0]])
            kpts /= kpts.new_tensor(scale_factor).repeat(
                (1, self.num_keypoints, 1))

        keep_idxs_nms = keep_indices_nms[0]
        pred_bbox = pred_bbox[keep_idxs_nms]
        pred_label = pred_label[keep_idxs_nms]
        pred_score = pred_score[keep_idxs_nms]
        kpts = kpts[keep_idxs_nms]
        kpts_vis = kpts_vis[keep_idxs_nms]

        pred_keypoints = kpts
        pred_keypoint_scores = kpts_vis

        result_list.append([pred_bbox, pred_label, pred_score, pred_keypoints, pred_keypoint_scores])

    return result_list
