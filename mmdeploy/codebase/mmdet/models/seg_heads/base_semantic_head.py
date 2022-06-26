# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.seg_heads.base_semantic_head.BaseSemanticHead.simple_test')
def base_semantic_head__simple_test(ctx, self, x, img_metas, **kwargs):
    """Rewrite `simple_test` for default backend.
    Support configured dynamic/static shape for model input and return
    semantic-segmentation result as Tensor instead of numpy array.
    Args:
        ctx (ContextCaller): The context with additional information.
        self: The instance of the original class.
        img (Tensor | List[Tensor]): Input image tensor(s).
        img_meta (list[dict]): Dict containing image's meta information
            such as `img_shape`.

    Returns:
        Tensor: `semseg` of shape [N, num_sem_class, H, W]
    """
    output = self.forward(x)
    seg_preds = output['seg_preds']

    h, w = img_metas[0]['img_shape'][:2]
    seg_preds = F.interpolate(
        seg_preds, size=(h, w), mode='bilinear', align_corners=False)
    return seg_preds
