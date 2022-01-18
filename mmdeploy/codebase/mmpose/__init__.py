# Copyright (c) OpenMMLab. All rights reserved.
from .deploy import MMPose, PoseDetection
from .models import *  # noqa: F401,F403

__all__ = ['MMPose', 'PoseDetection']
