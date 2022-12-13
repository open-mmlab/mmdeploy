# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.heatmap_heads.CPMHead.forward')
@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.heatmap_heads.MSPNHead.forward')
def mspn_head__forward(self, feats):
    """Rewrite `forward` of MSPNHead and CPMHead for default backend.

    1. return last stage heatmaps directly.

    Args:
        feats (tuple[Tensor]): Input features.

    Returns:
        output_heatmap (torch.Tensor): Output heatmaps.
    """
    ctx = FUNCTION_REWRITER.get_context()
    msmu_batch_heatmaps = ctx.origin_func(self, feats)
    batch_heatmaps = msmu_batch_heatmaps[-1]
    return batch_heatmaps
