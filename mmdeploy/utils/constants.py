from enum import Enum


class AdvancedEnum(Enum):

    @classmethod
    def get(cls, str, a):
        for k in cls:
            if k.value == str:
                return k
        return a


class Task(AdvancedEnum):
    TEXT_DETECTION = 'TextDetection'
    TEXT_RECOGNITION = 'TextRecognition'
    SEGMENTATION = 'Segmentation'
    SUPER_RESOLUTION = 'SuperResolution'
    CLASSIFICATION = 'Classification'
    OBJECT_DETECTION = 'ObjectDetection'


class Codebase(AdvancedEnum):
    MMDET = 'mmdet'
    MMSEG = 'mmseg'
    MMCLS = 'mmcls'
    MMOCR = 'mmocr'
    MMEDIT = 'mmedit'


class Backend(AdvancedEnum):
    PYTORCH = 'pytorch'
    TENSORRT = 'tensorrt'
    ONNXRUNTIME = 'onnxruntime'
    PPL = 'ppl'
    NCNN = 'ncnn'
    DEFAULT = 'default'
