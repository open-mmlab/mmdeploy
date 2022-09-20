# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.task_modules.coders.yolov5_bbox_coder'
    'YOLOv5BBoxCoder.decode',  # DeltaXYWHBBoxCoder.decode
    backend='default')
def yolov5bboxcoder__decode(
        ctx,
        self,
        priors,  # priors # bboxes
        pred_bboxes,  # pred_bboxes # pred_bboxes
        stride,  # stride # max_shape
):

    assert pred_bboxes.size(0) == priors.size(0)
    if pred_bboxes.ndim == 3:
        assert pred_bboxes.size(1) == priors.size(1)
    decoded_bboxes = delta2bbox(priors, pred_bboxes, stride)
    return decoded_bboxes


def delta2bbox(priors, pred_bboxes, stride):

    def xyxy2xywh(bbox_corner):
        matrix_center = torch.tensor([[0.5, 0., -1., 0.], [0., 0.5, 0., -1.],
                                      [0.5, 0., 1., 0.], [0., 0.5, 0., 1.]],
                                     dtype=torch.float32)
        return bbox_corner @ matrix_center.to(bbox_corner.device)

    x_center, y_center, w, h = xyxy2xywh(priors).unbind(2)

    # The anchor of mmdetection has been offset by 0.5
    x_center_pred = (pred_bboxes[..., 0] - 0.5) * 2 * stride + x_center
    y_center_pred = (pred_bboxes[..., 1] - 0.5) * 2 * stride + y_center
    w_pred = (pred_bboxes[..., 2] * 2)**2 * w
    h_pred = (pred_bboxes[..., 3] * 2)**2 * h

    decoded_bboxes = torch.stack(
        (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
         x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
        dim=-1)
    return decoded_bboxes
