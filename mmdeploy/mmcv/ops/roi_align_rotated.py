# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from torch import Tensor

from mmdeploy.core import SYMBOLIC_REWRITER


# Here using mmcv.ops.roi_align_rotated.__self__ to find
# mmcv.ops.roi_align.RoIAlignRotatedFunction, because RoIAlignRotatedFunction
# is not visible in mmcv.
@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.roi_align_rotated.__self__', backend='default')
def roi_align_rotated_default(g, input: Tensor, rois: Tensor,
                              output_size: List[int], spatial_scale: float,
                              sampling_ratio: int, aligned: bool,
                              clockwise: bool):
    """Rewrite symbolic function for default backend.

    Replace onnx::RoIAlignRotated with mmdeploy::MMCVRoIAlignRotated.

    Args:
        ctx (ContextCaller): The context with additional information.
        g (Graph): The traced onnx graph.
        input (Tensor): Input tensor, 4-D feature map of shape (N, C, H, W).
        rois (Tensor): Bx5 boxes. First column is the index into N. The other
            4 columns are xyxy.
        output_size(List[int]): Output size of height and width.
        spatial_scale (float):
        sampling_ratio (int): Number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        aligned (bool): With `aligned=True`, we first appropriately scale
            the ROI and then shift it by -0.5 prior to calling roi_align.
            This produces the correct neighbors;
        clockwise (bool): If True, the angle in each proposal follows a
            clockwise fashion in image space, otherwise, the angle is
            counterclockwise. Default: False.

    Returns:
        MMCVRoiAlign op for onnx.
    """
    return g.op(
        'mmdeploy::MMCVRoIAlignRotated',
        input,
        rois,
        output_height_i=output_size[0],
        output_width_i=output_size[1],
        spatial_scale_f=spatial_scale,
        sampling_ratio_i=sampling_ratio,
        aligned_i=aligned,
        clockwise_i=clockwise)
