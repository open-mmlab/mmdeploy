# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

import mmdeploy
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops import ONNXNMSop, TRTBatchedNMSop
from mmdeploy.utils import is_dynamic_batch


def select_nms_index(scores: torch.Tensor,
                     boxes: torch.Tensor,
                     nms_index: torch.Tensor,
                     batch_size: int,
                     keep_top_k: int = -1):
    """Transform NMS output.

    Args:
        scores (Tensor): The detection scores of shape
            [N, num_classes, num_boxes].
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        nms_index (Tensor): NMS output of bounding boxes indexing.
        batch_size (int): Batch size of the input image.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
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
    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))),
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
        device=topk_inds.device).view(-1, 1).expand_as(topk_inds)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]

    # slice and recover the tensor
    return batched_dets, batched_labels


def _multiclass_nms(boxes: Tensor,
                    scores: Tensor,
                    max_output_boxes_per_class: int = 1000,
                    iou_threshold: float = 0.5,
                    score_threshold: float = 0.05,
                    pre_top_k: int = -1,
                    keep_top_k: int = -1):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op. It
    only supports class-agnostic detection results. That is, the scores is of
    shape (N, num_bboxes, num_classes) and the boxes is of shape (N, num_boxes,
    4).
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]

    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        batch_inds = torch.arange(batch_size).view(
            -1, 1).expand_as(topk_inds).long()
        boxes = boxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSop.apply(boxes, scores,
                                       max_output_boxes_per_class,
                                       iou_threshold, score_threshold)

    dets, labels = select_nms_index(
        scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)

    return dets, labels


def _multiclass_nms_single(boxes: Tensor,
                           scores: Tensor,
                           max_output_boxes_per_class: int = 1000,
                           iou_threshold: float = 0.5,
                           score_threshold: float = 0.05,
                           pre_top_k: int = -1,
                           keep_top_k: int = -1):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    Single batch nms could be optimized.
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
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

    return dets, labels


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.core.post_processing._multiclass_nms')
def multiclass_nms__default(ctx,
                            boxes: Tensor,
                            scores: Tensor,
                            max_output_boxes_per_class: int = 1000,
                            iou_threshold: float = 0.5,
                            score_threshold: float = 0.05,
                            pre_top_k: int = -1,
                            keep_top_k: int = -1):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op.
    It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 4).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
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
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    deploy_cfg = ctx.cfg
    batch_size = boxes.size(0)
    if not is_dynamic_batch(deploy_cfg) and batch_size != 1:
        return _multiclass_nms_single(
            boxes,
            scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)
    else:
        return _multiclass_nms(
            boxes,
            scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.core.post_processing._multiclass_nms',
    backend='tensorrt')
def multiclass_nms_static(ctx,
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
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
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
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    keep_top_k = max_output_boxes_per_class if keep_top_k < 0 else min(
        max_output_boxes_per_class, keep_top_k)
    dets, labels = TRTBatchedNMSop.apply(boxes, scores, int(scores.shape[-1]),
                                         pre_top_k, keep_top_k, iou_threshold,
                                         score_threshold, -1)

    return dets, labels


@mark('multiclass_nms', inputs=['boxes', 'scores'], outputs=['dets', 'labels'])
def multiclass_nms(*args, **kwargs):
    """Wrapper function for `_multiclass_nms`."""
    return mmdeploy.codebase.mmdet.core.post_processing._multiclass_nms(
        *args, **kwargs)
