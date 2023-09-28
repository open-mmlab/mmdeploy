# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.codebase.mmpose.codecs import get_simcc_maximum
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import get_codebase_config


@FUNCTION_REWRITER.register_rewriter('mmpose.models.heads.RTMCCHead.forward')
@FUNCTION_REWRITER.register_rewriter('mmpose.models.heads.SimCCHead.forward')
def simcc_head__forward(self, feats):
    """Rewrite `forward` of SimCCHead for default backend.

    Args:
        feats (tuple[Tensor]): Input features.
    Returns:
        key-points (torch.Tensor): Output keypoints in
            shape of (N, K, 3)
    """
    ctx = FUNCTION_REWRITER.get_context()
    simcc_x, simcc_y = ctx.origin_func(self, feats)
    codebase_cfg = get_codebase_config(ctx.cfg)
    export_postprocess = codebase_cfg.get('export_postprocess', False)
    if not export_postprocess:
        return simcc_x, simcc_y
    assert self.decoder.use_dark is False, \
        'Do not support SimCCLabel with use_dark=True'
    pts, scores = get_simcc_maximum(simcc_x, simcc_y)
    pts /= self.decoder.simcc_split_ratio
    return pts, scores
