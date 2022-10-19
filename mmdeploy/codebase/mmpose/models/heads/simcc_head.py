# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.heatmap_heads.SimCCHead.predict')
def simcc_head__predict(ctx, self, feats, batch_data_samples, test_cfg=None):
    """Rewrite `predict` of HeatmapHead for default backend.

    1. skip decoding and return output tensor directly.

    Args:
        feats (tuple[Tensor]): Input features.
        batch_data_samples (list[SampleList]): Data samples contain
            image meta information.
        test_cfg (ConfigType): test config.

    Returns:
        output_heatmap (torch.Tensor): Output heatmaps.
    """
    simcc_x, simcc_y = self.forward(feats)
    return simcc_x, simcc_y
