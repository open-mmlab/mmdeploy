# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.ops import resize

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.decode_heads.decode_head.'
    'BaseDecodeHead.predict_by_feat')
def base_decode_head__predict_by_feat(ctx, self, seg_logits, batch_img_metas):
    """Rewrite `predict_by_feat` for default backend.

    Do not convert prediction to list

    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        seg_logits (Tensor): The output from decode head forward function.
        batch_img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.

    Returns:
        torch.Tensor: Output segmentation logits map.
    """
    seg_logits = resize(
        input=seg_logits,
        size=batch_img_metas[0]['img_shape'],
        mode='bilinear',
        align_corners=self.align_corners)
    return seg_logits
