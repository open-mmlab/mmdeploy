# Copyright (c) OpenMMLab. All rights reserved.
from .base import base_segmentor__forward
from .cascade_encoder_decoder import cascade_encoder_decoder__predict
from .encoder_decoder import encoder_decoder__predict

__all__ = [
    'base_segmentor__forward', 'encoder_decoder__predict',
    'cascade_encoder_decoder__predict'
]
