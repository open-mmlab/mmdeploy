# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import Tensor

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend, get_backend, get_ir_config


# Here using mmcv.ops.roi_align.__self__ to find
# mmcv.ops.roi_align.RoIAlignFunction, because RoIAlignFunction is not
# visible in mmcv.
@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.roi_align.__self__', backend='default')
def roi_align_default(g, input: Tensor, rois: Tensor, output_size: List[int],
                      spatial_scale: float, sampling_ratio: int,
                      pool_mode: str, aligned: bool):
    """Rewrite symbolic function for default backend.

    Replace onnx::RoiAlign with mmcv::MMCVRoiAlign for PPLNN. For ONNXRuntime,
    align operation get done outside the inference engine for opset versions
    lower than 16. By default,  onnx::RoiAlign get replaced to
    mmdeploy::MMCVRoiAlign.

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
    ctx = SYMBOLIC_REWRITER.get_context()
    backend = get_backend(ctx.cfg)
    if backend == Backend.PPLNN or backend == Backend.TENSORRT:
        domain = 'mmcv'
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
    else:
        from torch.onnx.symbolic_opset9 import _cast_Long
        from torch.onnx.symbolic_opset11 import add, select
        batch_indices = _cast_Long(
            g,
            g.op(
                'Squeeze',
                select(
                    g, rois, 1,
                    g.op(
                        'Constant',
                        value_t=torch.tensor([0], dtype=torch.long))),
                axes_i=[1]), False)
        rois = select(
            g, rois, 1,
            g.op(
                'Constant',
                value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        ir_cfg = get_ir_config(ctx.cfg)
        opset_version = ir_cfg.get('opset_version', 11)
        if opset_version < 16:
            # preprocess rois to make compatible with opset 16-
            # as for opset 16+, `aligned` get implemented inside onnxruntime.
            if aligned is True:
                rois = add(
                    g, rois,
                    g.op(
                        'Constant',
                        value_t=torch.tensor([-0.5 / spatial_scale],
                                             dtype=torch.float)))
            return g.op(
                'RoiAlign',
                input,
                rois,
                batch_indices,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=sampling_ratio,
                mode_s=pool_mode)
        else:
            return g.op(
                'RoiAlign',
                input,
                rois,
                batch_indices,
                output_height_i=output_size[0],
                output_width_i=output_size[1],
                spatial_scale_f=spatial_scale,
                sampling_ratio_i=sampling_ratio,
                mode_s=pool_mode,
                aligned_i=aligned)
