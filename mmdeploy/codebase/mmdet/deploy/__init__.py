# Copyright (c) OpenMMLab. All rights reserved.
from .object_detection_task import ObjectDetectionTask
from .utils import (clip_bboxes, gather_topk, get_post_processing_params,
                    pad_with_value, pad_with_value_if_necessary)

ObjectDetection = ObjectDetectionTask  # alias for mmyolo
__all__ = [
    'get_post_processing_params', 'clip_bboxes', 'pad_with_value',
    'pad_with_value_if_necessary', 'gather_topk', 'MMDetection',
    'ObjectDetectionTask', 'ObjectDetection'
]
