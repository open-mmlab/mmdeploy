# Copyright (c) OpenMMLab. All rights reserved.
from . import models  # noqa: F401,F403
from .deploy import MMPose, PoseDetection

__all__ = ['MMPose', 'PoseDetection']
