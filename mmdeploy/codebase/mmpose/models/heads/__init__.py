# Copyright (c) OpenMMLab. All rights reserved.
from .deeppose_regression_head import deeppose_regression_head__inference_model
from .topdown_heatmap_multi_stage_head import \
    topdown_heatmap_msmu_head__inference_model
from .topdown_heatmap_simple_head import \
    topdown_heatmap_simple_head__inference_model
from .vipnas_heatmap_simple_head import \
    vipnas_heatmap_simple_head__inference_model

__all__ = [
    'topdown_heatmap_simple_head__inference_model',
    'topdown_heatmap_msmu_head__inference_model',
    'deeppose_regression_head__inference_model',
    'vipnas_heatmap_simple_head__inference_model'
]
