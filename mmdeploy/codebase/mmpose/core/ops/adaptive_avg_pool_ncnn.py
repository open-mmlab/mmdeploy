# Copyright (c) OpenMMLab. All rights reserved.
import torch


class NcnnAdaptiveAvgPoolingOp(torch.autograd.Function):
    """Create AdaptiveAvgPooling op for ncnn.

    A dummy AdaptiveAvgPooling operator for ncnn end2end deployment.
    It will map to the AdaptiveAvgPooling op of ncnn. After converting
    to ncnn, AdaptiveAvgPooling op of ncnn will get called
    automatically.

    Args:
        loc (Tensor): The predicted boxes location tensor.
        conf (Tensor): The predicted boxes confidence of
            num_classes.
        anchor (Tensor): The prior anchors.
        score_threshold (float): Threshold of object
            score.
            Default: 0.35.
        nms_threshold (float): IoU threshold for NMS.
            Default: 0.45.
        nms_top_k (int): Number of bboxes after NMS.
            Default: 100.
        keep_top_k (int): Max number of bboxes of detection result.
            Default: 100.
        num_class (int): Number of classes, includes the background
            class.
            Default: 81.
    """

    @staticmethod
    def symbolic(g,
                 x,
                 output_size):
        """Symbolic function of dummy onnx AdaptiveAvgPooling op for ncnn."""
        return g.op(
            'mmdeploy::adaptive_avg_pool2d',
            x,
            output_size,
            outputs=1)

    @staticmethod
    def forward(ctx,
                x,
                output_size):
        """Forward function of dummy onnx DetectionOutput op for ncnn."""
        return torch.rand(x.shape[0], x.shape[1], output_size[0], output_size[1])


ncnn_adaptive_avg_pool_forward = NcnnAdaptiveAvgPoolingOp.apply
