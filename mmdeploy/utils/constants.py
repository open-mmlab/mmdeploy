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
    VOXEL_DETECTION = 'VoxelDetection'
    POSE_DETECTION = 'PoseDetection'
    ROTATED_DETECTION = 'RotatedDetection'
    VIDEO_RECOGNITION = 'VideoRecognition'
    ModelCompress = 'ModelCompress'
    MONO_DETECTION = 'MonoDetection'


class Codebase(AdvancedEnum):
    """Define codebase enumerations."""
    MMDET = 'mmdet'
    MMSEG = 'mmseg'
    MMPRETRAIN = 'mmpretrain'
    MMOCR = 'mmocr'
    MMAGIC = 'mmagic'
    MMDET3D = 'mmdet3d'
    MMPOSE = 'mmpose'
    MMROTATE = 'mmrotate'
    MMACTION = 'mmaction'
    MMRAZOR = 'mmrazor'
    MMYOLO = 'mmyolo'


class IR(AdvancedEnum):
    """Define intermediate representation enumerations."""
    ONNX = 'onnx'
    TORCHSCRIPT = 'torchscript'
    DEFAULT = 'default'


class Backend(AdvancedEnum):
    """Define backend enumerations."""
    PYTORCH = 'pytorch'
    TENSORRT = 'tensorrt'
    ONNXRUNTIME = 'onnxruntime'
    PPLNN = 'pplnn'
    NCNN = 'ncnn'
    SNPE = 'snpe'
    OPENVINO = 'openvino'
    SDK = 'sdk'
    TORCHSCRIPT = 'torchscript'
    RKNN = 'rknn'
    ASCEND = 'ascend'
    COREML = 'coreml'
    TVM = 'tvm'
    VACC = 'vacc'
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
    dict(component='CTCConvertor', cls_name='TextRecognizer'),
    Task.POSE_DETECTION:
    dict(component='Detector', cls_name='PoseDetector'),
    Task.ROTATED_DETECTION:
    dict(component='ResizeRBBox', cls_name='RotatedDetector'),
    Task.VIDEO_RECOGNITION:
    dict(component='BaseHead', cls_name='VideoRecognizer')
}

TENSORRT_MAX_TOPK = 3840
