# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.TopdownHeatmapSimpleHead.inference_model')
def top_down_heatmap_simple_head__inference_model(ctx,
                                                  self,
                                                  x,
                                                  flip_pairs=None):
    """Rewrite `forward_test` of TopDown for default backend."""
    assert flip_pairs is None
    output = self.forward(x)
    return output
