# Copyright (c) OpenMMLab. All rights reserved.

from .base import base_recognizer__forward
from .recognizer2d import recognizer2d__forward_test
from .recognizer3d import recognizer3d__forward_test

__all__ = ['base_recognizer__forward',
           'recognizer2d__forward_test', 'recognizer3d__forward_test']
