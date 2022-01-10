# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from torch.onnx import symbolic_helper as sym_help

from mmdeploy.core import SYMBOLIC_REWRITER


class ONNXNMSop(torch.autograd.Function):
    """Create onnx::NonMaxSuppression op.

    NMS in mmcv only supports one class with no batch info. This class assists
    in exporting NMS of ONNX's definition.
    """

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor,
                max_output_boxes_per_class: int, iou_threshold: float,
                score_threshold: float) -> Tensor:
        """Get NMS output indices.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            Tensor: Selected indices of boxes. 2-D tensor of shape
            (num_selected_indices, 3) with each row of
            [batch_index, class_index, box_index].
        """
        from mmcv.ops import nms
        batch_size, num_class, _ = scores.shape

        score_threshold = float(score_threshold)
        iou_threshold = float(iou_threshold)
        indices = []
        for batch_id in range(batch_size):
            for cls_id in range(num_class):
                _boxes = boxes[batch_id, ...]
                # score_threshold=0 requires scores to be contiguous
                _scores = scores[batch_id, cls_id, ...].contiguous()
                _, box_inds = nms(
                    _boxes,
                    _scores,
                    iou_threshold,
                    offset=0,
                    score_threshold=score_threshold,
                    max_num=max_output_boxes_per_class)
                batch_inds = torch.zeros_like(box_inds) + batch_id
                cls_inds = torch.zeros_like(box_inds) + cls_id
                indices.append(
                    torch.stack([batch_inds, cls_inds, box_inds], dim=-1))
        indices = torch.cat(indices)
        return indices

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor,
                 max_output_boxes_per_class: int, iou_threshold: float,
                 score_threshold: float):
        """Symbolic function for onnx::NonMaxSuppression.

        Args:
            g (Graph): The traced onnx graph.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            NonMaxSuppression op for onnx.
        """
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            outputs=1)


@SYMBOLIC_REWRITER.register_symbolic(
    'mmdeploy.mmcv.ops.ONNXNMSop', backend='default')
def nms_dynamic(ctx, g, boxes: Tensor, scores: Tensor,
                max_output_boxes_per_class: int, iou_threshold: float,
                score_threshold: float):
    """Rewrite symbolic function for default backend.

    Support max_output_boxes_per_class, iou_threshold, score_threshold of
    constant Tensor, which is aligned with ONNX's nms op.

    Args:
        ctx (ContextCaller): The context with additional information.
        g (Graph): The traced onnx graph.
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms.
        iou_threshold (float): IOU threshold of nms.
        score_threshold (float): score threshold of nms.

    Returns:
        NonMaxSuppression op for onnx.
    """

    if not sym_help._is_value(max_output_boxes_per_class):
        max_output_boxes_per_class = g.op(
            'Constant',
            value_t=torch.tensor(max_output_boxes_per_class, dtype=torch.long))

    if not sym_help._is_value(iou_threshold):
        iou_threshold = g.op(
            'Constant',
            value_t=torch.tensor([iou_threshold], dtype=torch.float))

    if not sym_help._is_value(score_threshold):
        score_threshold = g.op(
            'Constant',
            value_t=torch.tensor([score_threshold], dtype=torch.float))
    return g.op('NonMaxSuppression', boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold)


class TRTBatchedNMSop(torch.autograd.Function):
    """Create mmdeploy::TRTBatchedNMS op for TensorRT backend.

    NMS in ONNX supports dynamic outputs. This class helps replace
    onnx::NonMaxSuppression with mmdeploy::TRTBatchedNMS.
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
        """Forward of batched nms.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
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
            Tensor: Selected indices of boxes. 2-D tensor of shape
            (num_selected_indices, 3) with each row of
            [batch_index, class_index, box_index]. Note it is generated
            randomly to make it exportable to onnx.
        """
        batch_size, num_boxes, num_classes = scores.shape

        out_boxes = min(num_boxes, after_topk)
        return torch.rand(batch_size, out_boxes,
                          5).to(scores.device), torch.randint(
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
            'mmdeploy::TRTBatchedNMS',
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
