from enum import Enum


class Task(Enum):
    TEXT_DETECTION = 'TextDetection'
    TEXT_RECOGNITION = 'TextRecognition'
    SUPER_RESOLUTION = 'SuperResolution'


class Codebase(Enum):
    MMDET = 'mmdet'
    MMSEG = 'mmseg'
    MMCLS = 'mmcls'
    MMOCR = 'mmocr'
    MMEDIT = 'mmedit'


class Backend(Enum):
    PYTORCH = 'pytorch'
    TENSORRT = 'tensorrt'
    ONNXRUNTIME = 'onnxruntime'
    PPL = 'ppl'
    NCNN = 'ncnn'
    DEFAULT = 'default'
