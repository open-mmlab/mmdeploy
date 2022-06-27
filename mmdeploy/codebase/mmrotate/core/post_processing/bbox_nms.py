# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmrotate.core import obb2xyxy
from torch import Tensor

import mmdeploy
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops import (ONNXNMSop, ONNXNMSRotatedOp, TRTBatchedNMSop,
                               TRTBatchedRotatedNMSop)


def select_rnms_index(scores: torch.Tensor,
                      boxes: torch.Tensor,
                      nms_index: torch.Tensor,
                      batch_size: int,
                      keep_top_k: int = -1):
    """Transform NMSRotated output.

    Args:
        scores (Tensor): The detection scores of shape
            [N, num_classes, num_boxes].
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 6].
        nms_index (Tensor): NMS output of bounding boxes indexing.
        batch_size (int): Batch size of the input image.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 6]
            and `labels` of shape [N, num_det].
    """
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    # index by nms output
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)

    # batch all
    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(
        0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    batched_dets = batched_dets.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_dets.new_zeros(1))

    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where(
        (batch_inds == batch_template.unsqueeze(1)),
        batched_labels.new_ones(1) * -1)

    N = batched_dets.shape[0]

    # expand tensor to eliminate [0, ...] tensor
    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 6))),
                             1)
    batched_labels = torch.cat((batched_labels, batched_labels.new_zeros(
        (N, 1))), 1)

    # sort
    is_use_topk = keep_top_k > 0 and \
        (torch.onnx.is_in_onnx_export() or keep_top_k < batched_dets.shape[1])
    if is_use_topk:
        _, topk_inds = batched_dets[:, :, -1].topk(keep_top_k, dim=1)
    else:
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(
        batch_size, dtype=topk_inds.dtype,
        device=topk_inds.device).unsqueeze(1)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]

    # slice and recover the tensor
    return batched_dets, batched_labels


def _multiclass_nms_rotated(boxes: Tensor,
                            scores: Tensor,
                            iou_threshold: float = 0.1,
                            score_threshold: float = 0.05,
                            pre_top_k: int = -1,
                            keep_top_k: int = -1):
    """NMSRotated for multi-class bboxes.

    This function helps exporting to onnx with batch and multiclass NMSRotated
    op. It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 5).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 5].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): bbox threshold, bboxes with scores lower than
            it will not be considered.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 6]
            and `labels` of shape [N, num_det].
    """
    batch_size = scores.shape[0]

    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        batch_inds = torch.arange(batch_size).unsqueeze(1).long()
        boxes = boxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSRotatedOp.apply(boxes, scores, iou_threshold,
                                              score_threshold)

    dets, labels = select_rnms_index(
        scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)

    return dets, labels


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmrotate.core.post_processing.bbox_nms.'
    '_multiclass_nms_rotated',
    backend='tensorrt')
def multiclass_nms_rotated__tensorrt(ctx,
                                     boxes: Tensor,
                                     scores: Tensor,
                                     max_output_boxes_per_class: int = 1000,
                                     iou_threshold: float = 0.5,
                                     score_threshold: float = 0.05,
                                     pre_top_k: int = -1,
                                     keep_top_k: int = -1):
    """Wrapper for `multiclass_nms` with TensorRT.

    Args:
        ctx (ContextCaller): The context with additional information.
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 5].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 6]
            and `labels` of shape [N, num_det].
    """
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    keep_top_k = max_output_boxes_per_class if keep_top_k < 0 else min(
        max_output_boxes_per_class, keep_top_k)
    dets, labels = TRTBatchedRotatedNMSop.apply(boxes, scores,
                                                int(scores.shape[-1]),
                                                pre_top_k, keep_top_k,
                                                iou_threshold, score_threshold,
                                                -1)

    return dets, labels


@mark(
    'multiclass_nms_rotated',
    inputs=['boxes', 'scores'],
    outputs=['dets', 'labels'])
def multiclass_nms_rotated(*args, **kwargs):
    """Wrapper function for `_multiclass_nms`."""
    return mmdeploy.codebase.mmrotate.core.post_processing.bbox_nms.\
        _multiclass_nms_rotated(*args, **kwargs)


def _fake_multiclass_nms_rotated(boxes: Tensor,
                                 scores: Tensor,
                                 max_output_boxes_per_class: int = 1000,
                                 iou_threshold: float = 0.5,
                                 score_threshold: float = 0.0,
                                 pre_top_k: int = -1,
                                 keep_top_k: int = -1,
                                 version: str = 'le90'):
    """Fake NMSRotated for multi-class bboxes which use horizontal bboxes for
    NMS, but return the rotated bboxes result.

    This function helps exporting to onnx with batch and multiclass NMS op. It
    only supports class-agnostic detection results. That is, the scores is of
    shape (N, num_bboxes, num_classes) and the boxes is of shape (N, num_boxes,
    5).
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]

    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        batch_inds = torch.arange(batch_size).view(-1, 1).long()
        boxes = boxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]

    scores = scores.permute(0, 2, 1)
    hboxes = obb2xyxy(boxes, version)
    selected_indices = ONNXNMSop.apply(hboxes, scores,
                                       max_output_boxes_per_class,
                                       iou_threshold, score_threshold)

    dets, labels = select_rnms_index(
        scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)

    return dets, labels


@mark(
    'fake_multiclass_nms_rotated',
    inputs=['boxes', 'scores'],
    outputs=['dets', 'labels'])
def fake_multiclass_nms_rotated(*args, **kwargs):
    """Wrapper function for `_fake_multiclass_nms_rotated`."""
    return mmdeploy.codebase.mmrotate.core.post_processing.bbox_nms.\
        _fake_multiclass_nms_rotated(*args, **kwargs)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmrotate.core.post_processing.bbox_nms.'
    '_fake_multiclass_nms_rotated',
    backend='tensorrt')
def _fake_multiclass_nms_rotated__tensorrt(
        ctx,
        boxes: Tensor,
        scores: Tensor,
        max_output_boxes_per_class: int = 1000,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0,
        pre_top_k: int = -1,
        keep_top_k: int = -1,
        version: str = 'le90'):
    """Wrapper for `multiclass_nms` with TensorRT.

    Args:
        ctx (ContextCaller): The context with additional information.
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 5].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 6]
            and `labels` of shape [N, num_det].
    """
    batch_size = boxes.size(0)
    device = boxes.device
    hboxes = obb2xyxy(boxes, version)
    hboxes = hboxes if hboxes.dim() == 4 else hboxes.unsqueeze(2)
    keep_top_k = max_output_boxes_per_class if keep_top_k < 0 else min(
        max_output_boxes_per_class, keep_top_k)
    if pre_top_k > 512 * 10 or pre_top_k < 0:
        pre_top_k = 512 * 10

    dets, labels, index = TRTBatchedNMSop.apply(hboxes, scores,
                                                int(scores.shape[-1]),
                                                pre_top_k, keep_top_k,
                                                iou_threshold, score_threshold,
                                                -1, True)
    dets = torch.cat([boxes, scores], dim=-1)
    dets = torch.cat([dets, dets[:, :1, :] * 0], dim=1)
    batch_inds = torch.arange(batch_size, device=device).view(-1, 1)
    dets = dets[batch_inds, index, :]

    return dets, labels
