from enum import Enum


class AdvancedEnum(Enum):
    """Define an enumeration class."""

    @classmethod
    def get(cls, str, a):
        for k in cls:
            if k.value == str:
                return k
        return a


class Task(AdvancedEnum):
    """Define task enumerations."""
    TEXT_DETECTION = 'TextDetection'
    TEXT_RECOGNITION = 'TextRecognition'
    SEGMENTATION = 'Segmentation'
    SUPER_RESOLUTION = 'SuperResolution'
    CLASSIFICATION = 'Classification'
    OBJECT_DETECTION = 'ObjectDetection'


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
    PPL = 'ppl'
    NCNN = 'ncnn'
    OPENVINO = 'openvino'
    DEFAULT = 'default'
