# Copyright (c) OpenMMLab. All rights reserved.
from .fpn_cat import fpnc__forward__tensorrt
from .heads import base_text_det_head__predict, db_head__predict
from .single_stage_text_detector import single_stage_text_detector__forward

__all__ = [
    'fpnc__forward__tensorrt', 'base_text_det_head__predict',
    'single_stage_text_detector__forward', 'db_head__predict'
]
