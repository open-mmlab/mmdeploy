# Copyright (c) OpenMMLab. All rights reserved.
import torch


class NcnnDetectionOutputOp(torch.autograd.Function):
    """Create DetectionOutput op.

    A dummy DetectionOutput operator for ncnn end2end deployment.
    It will map to the DetectionOutput op of ncnn. After converting
    to ncnn, DetectionOutput op of ncnn will get called
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
                 loc,
                 conf,
                 anchor,
                 score_threshold=0.35,
                 nms_threshold=0.45,
                 nms_top_k=100,
                 keep_top_k=100,
                 num_class=81,
                 target_stds=[0.1, 0.1, 0.2, 0.2]):
        """Symbolic function of dummy onnx DetectionOutput op for ncnn."""
        return g.op(
            'mmdeploy::DetectionOutput',
            loc,
            conf,
            anchor,
            score_threshold_f=score_threshold,
            nms_threshold_f=nms_threshold,
            nms_top_k_i=nms_top_k,
            keep_top_k_i=keep_top_k,
            num_class_i=num_class,
            vars_f=target_stds,
            outputs=1)

    @staticmethod
    def forward(ctx,
                loc,
                conf,
                anchor,
                score_threshold=0.35,
                nms_threshold=0.45,
                nms_top_k=100,
                keep_top_k=100,
                num_class=81,
                target_stds=[0.1, 0.1, 0.2, 0.2]):
        """Forward function of dummy onnx DetectionOutput op for ncnn."""
        return torch.rand(1, 100, 6)


ncnn_detection_output_forward = NcnnDetectionOutputOp.apply
