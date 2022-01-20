# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.TopdownHeatmapMSMUHead.inference_model')
def top_down_heatmap_msmu_head__inference_model(ctx, self, x, flip_pairs=None):
    """Rewrite ``inference_model`` for default backend."""
    assert flip_pairs is None
    output = self.forward(x)
    assert isinstance(output, list)
    output = output[-1]
    return output
