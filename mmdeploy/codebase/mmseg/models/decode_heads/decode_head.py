# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER, mark


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.decode_heads.decode_head.BaseDecodeHead.cls_seg',
    backend='vacc')
def base_decode_head__cls_seg__vacc(self, feat):
    """Classify each pixel."""

    ctx = FUNCTION_REWRITER.get_context()
    feat = ctx.origin_func(self, feat)

    # mark seg_maps
    @mark('seg_maps', outputs=['output'])
    def __mark_feat(feat):
        return feat

    feat = __mark_feat(feat)

    return feat
