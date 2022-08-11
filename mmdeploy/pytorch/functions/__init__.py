# Copyright (c) OpenMMLab. All rights reserved.
from .adaptive_pool import (adaptive_avg_pool2d__default,
                            adaptive_avg_pool2d__ncnn)
from .atan2 import atan2__default
from .chunk import chunk__ncnn, chunk__torchscript
from .expand import expand__ncnn
from .getattribute import tensor__getattribute__ncnn
from .group_norm import group_norm__ncnn
from .interpolate import interpolate__ncnn, interpolate__tensorrt
from .linear import linear__ncnn
from .masked_fill import masked_fill__onnxruntime
from .normalize import normalize__ncnn
from .repeat import tensor__repeat__tensorrt
from .size import tensor__size__ncnn
from .tensor_setitem import tensor__setitem__default
from .topk import topk__dynamic, topk__tensorrt
from .triu import triu__default

__all__ = [
    'tensor__getattribute__ncnn', 'group_norm__ncnn', 'interpolate__ncnn',
    'interpolate__tensorrt', 'linear__ncnn', 'tensor__repeat__tensorrt',
    'tensor__size__ncnn', 'topk__dynamic', 'topk__tensorrt', 'chunk__ncnn',
    'triu__default', 'atan2__default', 'normalize__ncnn', 'expand__ncnn',
    'chunk__torchscript', 'masked_fill__onnxruntime',
    'tensor__setitem__default', 'adaptive_avg_pool2d__default',
    'adaptive_avg_pool2d__ncnn'
]
