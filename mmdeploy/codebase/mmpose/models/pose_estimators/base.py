# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.pose_estimators.base.BasePoseEstimator.forward')
def base_pose_estimator__forward(self, inputs, *args, **kwargs):
    """Rewrite `forward` of TopDown for default backend.'.

    1.directly call _forward of subclass.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (BasePoseEstimator): The instance of the class Object
            BasePoseEstimator.
        inputs (torch.Tensor[NxCxHxW]): Input images.

    Returns:
        torch.Tensor: The predicted heatmaps.
    """
    return self._forward(inputs)
