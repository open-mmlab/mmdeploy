# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.pose_estimators.base.BasePoseEstimator.forward')
def base_pose_estimator__forward(ctx, self, inputs, *args, **kwargs):
    """Rewrite `forward` of TopDown for default backend.'.

    1.directly call _forward of subclass.

    Args:
        inputs (torch.Tensor[NxCxHxW]): Input images.

    Returns:
        torch.Tensor: The predicted heatmaps.
    """
    return self._forward(inputs)
