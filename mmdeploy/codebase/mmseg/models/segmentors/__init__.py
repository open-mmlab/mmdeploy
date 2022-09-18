# Copyright (c) OpenMMLab. All rights reserved.
from .base import base_segmentor__forward
from .encoder_decoder import (encoder_decoder__simple_test,
                              encoder_decoder__simple_test__rknn)

__all__ = [
    'base_segmentor__forward', 'encoder_decoder__simple_test',
    'encoder_decoder__simple_test__rknn'
]
