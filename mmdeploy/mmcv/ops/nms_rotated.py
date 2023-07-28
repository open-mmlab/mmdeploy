# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

import mmdeploy
from mmdeploy.core import FUNCTION_REWRITER, mark


class ONNXNMSRotatedOp(torch.autograd.Function):
    """Create onnx::NMSRotated op."""

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor, iou_threshold: float,
                score_threshold: float) -> Tensor:
        """Get NMS rotated output indices.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 5].
            scores (Tensor): The detection scores of shape
                [N, num_classes, num_boxes].
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): bbox threshold, bboxes with scores
            lower than it will not be considered.

        Returns:
            Tensor: Selected indices of boxes.
        """
        from mmcv.ops import nms_rotated
        batch_size, num_class, _ = scores.shape

        indices = []
        for batch_id in range(batch_size):
            for cls_id in range(num_class):
                _boxes = boxes[batch_id, ...]
                # score_threshold=0 requires scores to be contiguous
                _scores = scores[batch_id, cls_id, ...].contiguous()
                valid_mask = _scores > score_threshold
                _boxes, _scores = _boxes[valid_mask], _scores[valid_mask]
                if _boxes.shape[0] == 0:
                    continue
                valid_inds = torch.nonzero(
                    valid_mask, as_tuple=False).squeeze(dim=1)
                _, box_inds = nms_rotated(
                    _boxes, _scores, iou_threshold=iou_threshold)
                box_inds = valid_inds[box_inds]
                batch_inds = torch.zeros_like(box_inds) + batch_id
                cls_inds = torch.zeros_like(box_inds) + cls_id
                indices.append(
                    torch.stack([batch_inds, cls_inds, box_inds], dim=-1))

        indices = torch.cat(indices) if len(indices) > 0 else torch.zeros(
            (0, 3), dtype=torch.long, device=boxes.device)
        return indices

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor, iou_threshold: float,
                 score_threshold: float):
        """Symbolic function for onnx::NMSRotated.

        Args:
            g (Graph): The traced onnx graph.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): bbox threshold, bboxes with scores
            lower than it will not be considered.

        Returns:
            NMSRotated op for onnx.
        """
        return g.op(
            'mmdeploy::NMSRotated',
            boxes,
            scores,
            iou_threshold_f=float(iou_threshold),
            score_threshold_f=float(score_threshold))


class TRTBatchedRotatedNMSop(torch.autograd.Function):
    """Create mmdeploy::TRTBatchedRotatedNMSop op for TensorRT backend.

    NMS in ONNX supports dynamic outputs. This class helps replace
    onnx::NonMaxSuppression with mmdeploy::TRTBatchedRotatedNMSop.
    """

    @staticmethod
    def forward(ctx,
                boxes: Tensor,
                scores: Tensor,
                num_classes: int,
                pre_topk: int,
                after_topk: int,
                iou_threshold: float,
                score_threshold: float,
                background_label_id: int = -1):
        """Forward of batched rotated nms.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 5].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            num_classes (int): MThe number of classes in the network.
            pre_topk (int): The number of bounding boxes to be fed into
                the NMS step.
            after_topk (int): The number of total bounding boxes to be kept
                per-image after the NMS step. Should be less than or equal
                to the pre_topk value.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.
            background_label_id (int): The label ID for the background class.
                If there is no background class, set it to -1.

        Returns:
            dets (Tensor): Bboxes and scores of the rotated nms results.
            labels (Tensor): Class id of the rotated nms results.
        """
        batch_size, num_boxes, num_classes = scores.shape

        out_boxes = min(num_boxes, after_topk)
        return torch.rand(batch_size, out_boxes,
                          6).to(scores.device), torch.randint(
                              0, num_classes,
                              (batch_size, out_boxes)).to(scores.device)

    @staticmethod
    def symbolic(g,
                 boxes: Tensor,
                 scores: Tensor,
                 num_classes: int,
                 pre_topk: int,
                 after_topk: int,
                 iou_threshold: float,
                 score_threshold: float,
                 background_label_id: int = -1):
        """Symbolic function for mmdeploy::TRTBatchedNMS."""
        return g.op(
            'mmdeploy::TRTBatchedRotatedNMS',
            boxes,
            scores,
            num_classes_i=num_classes,
            background_label_id_i=background_label_id,
            iou_threshold_f=iou_threshold,
            score_threshold_f=score_threshold,
            topk_i=pre_topk,
            keep_topk_i=after_topk,
            is_normalized_i=False,
            clip_boxes_i=False,
            outputs=2)


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
                            max_output_boxes_per_class: int = 1000,
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
    func_name='mmdeploy.mmcv.ops.nms_rotated._multiclass_nms_rotated',
    backend='tensorrt')
def multiclass_nms_rotated__tensorrt(boxes: Tensor,
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
def multiclass_nms_rotated(boxes: Tensor,
                           scores: Tensor,
                           max_output_boxes_per_class: int = 1000,
                           iou_threshold: float = 0.1,
                           score_threshold: float = 0.05,
                           pre_top_k: int = -1,
                           keep_top_k: int = -1):
    """Wrapper function for `_multiclass_nms`."""
    return mmdeploy.mmcv.ops.nms_rotated._multiclass_nms_rotated(
        boxes,
        scores,
        max_output_boxes_per_class=max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)
