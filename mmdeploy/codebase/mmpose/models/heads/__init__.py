# Copyright (c) OpenMMLab. All rights reserved.
from .topdown_heatmap_multi_stage_head import \
    top_down_heatmap_msmu_head__inference_model
from .topdown_heatmap_simple_head import \
    top_down_heatmap_simple_head__inference_model

__all__ = [
    'top_down_heatmap_simple_head__inference_model',
    'top_down_heatmap_msmu_head__inference_model'
]
