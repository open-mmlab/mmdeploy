# Copyright (c) OpenMMLab. All rights reserved.
from .heatmap_head import heatmap_head__predict
from .mspn_head import mspn_head__predict
from .regression_head import regression_head__predict

__all__ = [
    'regression_head__predict', 'heatmap_head__predict', 'mspn_head__predict'
]
