# Copyright (c) OpenMMLab. All rights reserved.
from enum import Enum


class AdvancedEnum(Enum):
    """Define an enumeration class."""

    @classmethod
    def get(cls, value):
        """Get the key through a value."""
        for k in cls:
            if k.value == value:
                return k

        raise KeyError(f'Cannot get key by value "{value}" of {cls}')


class Task(AdvancedEnum):
    """Define task enumerations."""
    TEXT_DETECTION = 'TextDetection'
    TEXT_RECOGNITION = 'TextRecognition'
    SEGMENTATION = 'Segmentation'
    SUPER_RESOLUTION = 'SuperResolution'
    CLASSIFICATION = 'Classification'
    OBJECT_DETECTION = 'ObjectDetection'
    INSTANCE_SEGMENTATION = 'InstanceSegmentation'


class Codebase(AdvancedEnum):
    """Define codebase enumerations."""
    MMDET = 'mmdet'
    MMSEG = 'mmseg'
    MMCLS = 'mmcls'
    MMOCR = 'mmocr'
    MMEDIT = 'mmedit'


class Backend(AdvancedEnum):
    """Define backend enumerations."""
    PYTORCH = 'pytorch'
    TENSORRT = 'tensorrt'
    ONNXRUNTIME = 'onnxruntime'
    PPLNN = 'pplnn'
    NCNN = 'ncnn'
    OPENVINO = 'openvino'
    SDK = 'sdk'
    DEFAULT = 'default'


SDK_TASK_MAP = {
    Task.CLASSIFICATION:
    dict(component='LinearClsHead', cls_name='Classifier'),
    Task.OBJECT_DETECTION:
    dict(component='ResizeBBox', cls_name='Detector'),
    Task.INSTANCE_SEGMENTATION:
    dict(component='ResizeInstanceMask', cls_name='Detector'),
    Task.SEGMENTATION:
    dict(component='ResizeMask', cls_name='Segmentor'),
    Task.SUPER_RESOLUTION:
    dict(component='TensorToImg', cls_name='Restorer'),
    Task.TEXT_DETECTION:
    dict(component='TextDetHead', cls_name='TextDetector'),
    Task.TEXT_RECOGNITION:
    dict(component='CTCConvertor', cls_name='TextRecognizer')
}
