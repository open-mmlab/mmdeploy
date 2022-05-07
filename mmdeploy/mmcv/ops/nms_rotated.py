# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor


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
        from mmcv.utils import ext_loader
        ext_module = ext_loader.load_ext('_ext', ['nms_rotated'])
        batch_size, num_class, _ = scores.shape

        indices = []
        for batch_id in range(batch_size):
            for cls_id in range(num_class):
                _boxes = boxes[batch_id, ...]
                # score_threshold=0 requires scores to be contiguous
                _scores = scores[batch_id, cls_id, ...].contiguous()
                valid_mask = _scores > score_threshold
                _boxes, _scores = _boxes[valid_mask], _scores[valid_mask]
                valid_inds = torch.nonzero(
                    valid_mask, as_tuple=False).squeeze(dim=1)
                _, order = _scores.sort(0, descending=True)
                dets_sorted = _boxes.index_select(0, order)
                box_inds = ext_module.nms_rotated(_boxes, _scores, order,
                                                  dets_sorted, iou_threshold,
                                                  0)
                box_inds = valid_inds[box_inds]
                batch_inds = torch.zeros_like(box_inds) + batch_id
                cls_inds = torch.zeros_like(box_inds) + cls_id
                indices.append(
                    torch.stack([batch_inds, cls_inds, box_inds], dim=-1))

        indices = torch.cat(indices)
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
