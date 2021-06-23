import torch
from torch.onnx import symbolic_helper as sym_help

from mmdeploy.utils import SYMBOLICS_REGISTER


class DummyONNXNMSop(torch.autograd.Function):
    """DummyONNXNMSop.

    This class is only for creating onnx::NonMaxSuppression.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                score_threshold):

        return DummyONNXNMSop.output

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold,
                 score_threshold):
        return g.op(
            'NonMaxSuppression',
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            outputs=1)


@SYMBOLICS_REGISTER.register_symbolic(
    'mmdeploy.mmcv.ops.DummyONNXNMSop', backend='default')
def nms_default(symbolic_wrapper, g, boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold):
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


@SYMBOLICS_REGISTER.register_symbolic(
    'mmdeploy.mmcv.ops.DummyONNXNMSop', backend='tensorrt')
def nms_tensorrt(symbolic_wrapper, g, boxes, scores,
                 max_output_boxes_per_class, iou_threshold, score_threshold):
    if sym_help._is_value(max_output_boxes_per_class):
        max_output_boxes_per_class = sym_help._maybe_get_const(
            max_output_boxes_per_class, 'i')

    if sym_help._is_value(iou_threshold):
        iou_threshold = sym_help._maybe_get_const(iou_threshold, 'f')

    if sym_help._is_value(score_threshold):
        score_threshold = sym_help._maybe_get_const(score_threshold, 'f')

    return g.op(
        'mmcv::NonMaxSuppression',
        boxes,
        scores,
        max_output_boxes_per_class_i=max_output_boxes_per_class,
        iou_threshold_f=iou_threshold,
        score_threshold_f=score_threshold,
        center_point_box_i=0,
        offset_i=0)
