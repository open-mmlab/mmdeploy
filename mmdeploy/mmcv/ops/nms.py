import torch
from torch.onnx import symbolic_helper as sym_help

from mmdeploy.core import SYMBOLIC_REGISTER


class DummyONNXNMSop(torch.autograd.Function):
    """DummyONNXNMSop.

    This class is only for creating onnx::NonMaxSuppression.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_boxes_per_class, iou_threshold,
                score_threshold):
        batch_size, num_class, num_box = scores.shape
        # create dummy indices of nms output
        num_fake_det = 2
        batch_inds = torch.randint(batch_size, (num_fake_det, 1))
        cls_inds = torch.randint(num_class, (num_fake_det, 1))
        box_inds = torch.randint(num_box, (num_fake_det, 1))
        indices = torch.cat([batch_inds, cls_inds, box_inds], dim=1)
        return indices.to(scores.device)

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


@SYMBOLIC_REGISTER.register_symbolic(
    'mmdeploy.mmcv.ops.DummyONNXNMSop', backend='default')
def nms_dynamic(ctx, g, boxes, scores, max_output_boxes_per_class,
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


@SYMBOLIC_REGISTER.register_symbolic(
    'mmdeploy.mmcv.ops.DummyONNXNMSop', backend='tensorrt')
def nms_static(ctx, g, boxes, scores, max_output_boxes_per_class,
               iou_threshold, score_threshold):
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


class TRTBatchedNMSop(torch.autograd.Function):

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
        batch_size, num_boxes, num_classes = scores.shape

        out_boxes = min(num_boxes, after_topk)
        return torch.rand(batch_size, out_boxes,
                          5).to(scores.device), torch.randint(
                              0, num_classes,
                              (batch_size, out_boxes)).to(scores.device)

    @staticmethod
    def symbolic(g, boxes, scores, num_classes, pre_topk, after_topk,
                 iou_threshold, score_threshold, background_label_id):
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
