# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.pose_estimators.topdown.TopdownPoseEstimator.predict')
def topdown_pose_estimator__predict(ctx, self, inputs, data_samples, **kwargs):
    """Rewrite `predict` of TopdownPoseEstimator for default backend.'.

    1. skip flip_test
    2. avoid call `add_pred_to_datasample`

    Args:
        inputs (torch.Tensor[NxCxHxW]): Input images.
        data_samples (SampleList | None): Data samples contain
            image meta information.

    Returns:
        torch.Tensor: The predicted heatmaps.
    """
    assert self.with_head, ('The model must have head to perform prediction.')
    feats = self.extract_feat(inputs)
    preds = self.head.predict(feats, data_samples, test_cfg=self.test_cfg)
    return preds
