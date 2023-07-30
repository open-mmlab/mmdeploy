# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.single_stage_mono3d.'
    'SingleStageMono3DDetector.forward')
def singlestagemono3ddetector__forward(self, inputs: list, **kwargs):
    """Rewrite this func to r.

    Args:
        inputs (dict): Input dict comprises `imgs`

    Returns:
        list: two torch.Tensor
    """
    x = self.extract_feat(inputs)
    results = self.bbox_head.forward(x)
    return results[0], results[1]
