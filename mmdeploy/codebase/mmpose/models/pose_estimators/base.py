# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.pose_estimators.base.BasePoseEstimator.forward')
def base_pose_estimator__forward(ctx,
                                 self,
                                 inputs,
                                 data_samples=None,
                                 mode='predict',
                                 **kwargs):
    """Rewrite `forward_test` of TopDown for default backend.'.

    1. only support mode='predict'.
    2. create data_samples if necessary

    Args:
        inputs (torch.Tensor[NxCxHxW]): Input images.
        data_samples (SampleList | None): Data samples contain
            image meta information.

    Returns:
        torch.Tensor: The predicted heatmaps.
    """
    if data_samples is None:
        from mmpose.structures import PoseDataSample
        _, c, h, w = [int(_) for _ in inputs.shape]
        metainfo = dict(
            img_shape=(h, w, c),
            input_size=(w, h),
            heatmap_size=self.cfg.codec.heatmap_size)
        data_sample = PoseDataSample(metainfo=metainfo)
        data_samples = [data_sample]

    return self.predict(inputs, data_samples)
