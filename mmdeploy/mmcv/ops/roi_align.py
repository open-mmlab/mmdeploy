# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from torch import Tensor

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend, get_backend


# Here using mmcv.ops.roi_align.__self__ to find
# mmcv.ops.roi_align.RoIAlignFunction, because RoIAlignFunction is not
# visible in mmcv.
@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.roi_align.__self__', backend='default')
def roi_align_default(ctx, g, input: Tensor, rois: Tensor,
                      output_size: List[int], spatial_scale: float,
                      sampling_ratio: int, pool_mode: str, aligned: bool):
    """Rewrite symbolic function for default backend.

    Replace onnx::RoiAlign with mmdeploy::MMCVRoiAlign.

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
        pool_mode (str): Pooling mode in each bin, could be 'avg' or 'max'.
        aligned (bool): With `aligned=True`, we first appropriately scale
            the ROI and then shift it by -0.5 prior to calling roi_align.
            This produces the correct neighbors;

    Returns:
        MMCVRoiAlign op for onnx.
    """
    backend = get_backend(ctx.cfg)
    if backend == Backend.PPLNN:
        domain = 'mmcv'
    else:
        domain = 'mmdeploy'
    return g.op(
        f'{domain}::MMCVRoiAlign',
        input,
        rois,
        output_height_i=output_size[0],
        output_width_i=output_size[1],
        spatial_scale_f=spatial_scale,
        sampling_ratio_i=sampling_ratio,
        mode_s=pool_mode,
        aligned_i=aligned)
