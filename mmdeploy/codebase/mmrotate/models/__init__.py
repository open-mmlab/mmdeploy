# Copyright (c) OpenMMLab. All rights reserved.
from .rotated_anchor_head import rotated_anchor_head__get_bbox
from .single_stage_rotated_detector import \
    single_stage_rotated_detector__simple_test

__all__ = [
    'single_stage_rotated_detector__simple_test',
    'rotated_anchor_head__get_bbox'
]
