# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor


class ONNXNMSRotatedOp(torch.autograd.Function):
    """Create onnx::NMSRotated op."""

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor,
                iou_threshold: float) -> Tensor:
        """Get NMS rotated output indices.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            iou_threshold (float): IOU threshold of nms.

        Returns:
            Tensor: Selected indices of boxes.
        """
        from mmcv.utils import ext_loader
        ext_module = ext_loader.load_ext('_ext', ['nms_rotated'])

        _, order = scores.sort(0, descending=True)
        dets_sorted = boxes.index_select(0, order)
        keep_inds = ext_module.nms_rotated(boxes, scores, order, dets_sorted,
                                           iou_threshold, 0)
        return keep_inds

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor, iou_threshold: float):
        """Symbolic function for onnx::NMSRotated.

        Args:
            g (Graph): The traced onnx graph.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            iou_threshold (float): IOU threshold of nms.

        Returns:
            NMSRotated op for onnx.
        """
        return g.op(
            'mmdeploy::NMSRotated',
            boxes,
            scores,
            iou_threshold_f=float(iou_threshold))
