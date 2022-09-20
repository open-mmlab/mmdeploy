# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.heatmap_heads.MSPNHead.predict')
def mspn_head__predict(ctx, self, feats, batch_data_samples, test_cfg=None):
    """Rewrite `predict` of HeatmapHead for default backend.

    1. skip heatmaps decoding and return heatmaps directly.

    Args:
        feats (tuple[Tensor]): Input features.
        batch_data_samples (list[SampleList]): Data samples contain
            image meta information.
        test_cfg (ConfigType): test config.

    Returns:
        output_heatmap (torch.Tensor): Output heatmaps.
    """
    msmu_batch_heatmaps = self.forward(feats)
    batch_heatmaps = msmu_batch_heatmaps[-1]
    return batch_heatmaps
