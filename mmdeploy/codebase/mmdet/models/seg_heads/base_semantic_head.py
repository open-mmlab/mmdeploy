# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.seg_heads.base_semantic_head.BaseSemanticHead.predict')
def base_semantic_head__predict(self, x, batch_img_metas, rescale=False):
    """Rewrite `predict` for default backend. Support configured dynamic/static
    shape for model input and return semantic-segmentation result as Tensor
    instead of numpy array.

    Args:
        x (Union[Tensor, Tuple[Tensor]]): Feature maps.
        batch_img_metas (List[dict]): List of image information.
        rescale (bool): Whether to rescale the results.
            Defaults to False.

    Returns:
        Tensor: `semseg` of shape [N, num_sem_class, H, W]
    """
    seg_preds = self.forward(x)['seg_preds']
    img_shape = batch_img_metas[0]['batch_input_shape']
    seg_preds = F.interpolate(
        seg_preds,
        size=(img_shape[0], img_shape[1]),
        mode='bilinear',
        align_corners=False)
    return seg_preds
