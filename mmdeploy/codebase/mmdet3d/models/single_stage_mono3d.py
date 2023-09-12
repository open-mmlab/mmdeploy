# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.single_stage_mono3d.'
    'SingleStageMono3DDetector.forward')
def singlestagemono3ddetector__forward(self, inputs: Tensor, **kwargs):
    """Rewrite to support feed inputs of Tensor type.

    Args:
        inputs (Tensor): Input image

    Returns:
        list: two torch.Tensor
    """

    x = self.extract_feat({'imgs': inputs})
    results = self.bbox_head.forward(x)
    return results[0], results[1]
