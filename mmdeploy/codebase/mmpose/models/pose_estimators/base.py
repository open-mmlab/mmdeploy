# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.pose_estimators.base.BasePoseEstimator.forward')
def base_pose_estimator__forward(ctx,
                                 self,
                                 batch_inputs,
                                 batch_data_samples=None,
                                 mode='predict',
                                 **kwargs):
    """Rewrite `forward_test` of TopDown for default backend.'.

    1. only support mode='predict'.
    2. create batch_data_samples if necessary

    Args:
        batch_inputs (torch.Tensor[NxCxHxW]): Input images.
        batch_data_samples (SampleList | None): Data samples contain
            image meta information.

    Returns:
        torch.Tensor: The predicted heatmaps.
    """
    if batch_data_samples is None:
        from mmpose.core import PoseDataSample
        _, c, h, w = [int(_) for _ in batch_inputs.shape]
        metainfo = dict(
            img_shape=(h, w, c),
            crop_size=(h, w),
            heatmap_size=self.cfg.codec.heatmap_size)
        data_sample = PoseDataSample(metainfo=metainfo)
        batch_data_samples = [data_sample]

    return self.predict(batch_inputs, batch_data_samples)
