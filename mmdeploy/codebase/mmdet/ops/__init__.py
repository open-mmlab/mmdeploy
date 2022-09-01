# Copyright (c) OpenMMLab. All rights reserved.
from .detection_output import ncnn_detection_output_forward
from .prior_box import ncnn_prior_box_forward

__all__ = ['ncnn_detection_output_forward', 'ncnn_prior_box_forward']
