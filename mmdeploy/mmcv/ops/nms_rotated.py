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
                if _boxes.shape[0] == 0:
                    continue
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


class TRTBatchedBEVNMSop(torch.autograd.Function):
    """Create mmdeploy::TRTBatchedBEVNMS op for TensorRT backend.

    NMS in ONNX supports dynamic outputs. This class helps replace
    onnx::NonMaxSuppression with mmdeploy::TRTBatchedBEVNMS.
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
                background_label_id: int = -1,
                return_index: bool = True):
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
            dets (Tensor): Bboxes and scores of the bev nms results.
            labels (Tensor): Class id of the bev nms results.
            index (Tensor): Bboxes index of the bev nms results.
        """
        batch_size, num_boxes, num_classes = scores.shape

        out_boxes = min(num_boxes, after_topk)
        ret = (torch.rand(batch_size, out_boxes, 6).to(scores.device),
               torch.randint(0, num_classes,
                             (batch_size, out_boxes)).to(scores.device))
        if return_index:
            ret = ret + (torch.randint(
                0, out_boxes, (batch_size, out_boxes)).to(scores.device), )
        return ret

    @staticmethod
    def symbolic(g,
                 boxes: Tensor,
                 scores: Tensor,
                 num_classes: int,
                 pre_topk: int,
                 after_topk: int,
                 iou_threshold: float,
                 score_threshold: float,
                 background_label_id: int = -1,
                 return_index: bool = True):
        """Symbolic function for mmdeploy::TRTBatchedBEVNMS."""
        return g.op(
            'mmdeploy::TRTBatchedBEVNMS',
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
            return_index_i=return_index,
            outputs=3 if return_index else 2)
