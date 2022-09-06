# Copyright (c) OpenMMLab. All rights reserved.
from .mmdetection3d import MMDetection3d, VoxelDetection
# from .voxel_detection import 
from .voxel_detection_model import VoxelDetectionModel

__all__ = ['MMDetection3d', 'VoxelDetection', 'VoxelDetectionModel']
