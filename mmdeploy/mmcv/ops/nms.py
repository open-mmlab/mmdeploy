import torch
from mmcv.ops import nms
from torch.onnx import symbolic_helper as sym_help

from mmdeploy.core import SYMBOLIC_REWRITER


class ONNXNMSop(torch.autograd.Function):
    """Create onnx::NonMaxSuppression op."""

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                score_threshold):
        """Forward of onnx nms."""
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
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                 score_threshold):
        """Symbolic function for onnx::NonMaxSuppression."""
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
def nms_dynamic(ctx, g, boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold):
    """Rewrite symbolic function for default backend."""

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
    """Create mmcv::TRTBatchedNMS op for TensorRT backend."""

    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                num_classes,
                pre_topk,
                after_topk,
                iou_threshold,
                score_threshold,
                background_label_id=-1):
        """Forward of batched nms."""
        batch_size, num_boxes, num_classes = scores.shape

        out_boxes = min(num_boxes, after_topk)
        return torch.rand(batch_size, out_boxes,
                          5).to(scores.device), torch.randint(
                              0, num_classes,
                              (batch_size, out_boxes)).to(scores.device)

    @staticmethod
    def symbolic(g, boxes, scores, num_classes, pre_topk, after_topk,
                 iou_threshold, score_threshold, background_label_id):
        """Symbolic function for mmcv::TRTBatchedNMS."""
        return g.op(
            'mmcv::TRTBatchedNMS',
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
