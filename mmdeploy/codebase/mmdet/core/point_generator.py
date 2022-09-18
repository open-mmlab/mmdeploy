# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils.constants import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.core.anchor.MlvlPointGenerator.single_level_grid_priors',
    backend=Backend.TENSORRT.value)
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.core.anchor.MlvlPointGenerator.single_level_grid_priors',
    backend=Backend.RKNN.value)
def mlvl_point_generator__single_level_grid_priors__tensorrt(
        ctx,
        self,
        featmap_size,
        level_idx,
        dtype=torch.float32,
        device='cuda',
        with_stride=False):
    """Rewrite `single_level_grid_priors` of `MlvlPointGenerator` as
    onnx2tensorrt raise the error of shape inference for YOLOX with some
    versions of TensorRT.

    Args:
        featmap_size (tuple[int]): Size of the feature maps, arrange as
            (h, w).
        level_idx (int): The index of corresponding feature map level.
        dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
        device (str, optional): The device the tensor will be put on.
            Defaults to 'cuda'.
        with_stride (bool): Concatenate the stride to the last dimension
            of points.

    Return:
        Tensor: Points of single feature levels.
        The shape of tensor should be (N, 2) when with stride is
        ``False``, where N = width * height, width and height
        are the sizes of the corresponding feature level,
        and the last dimension 2 represent (coord_x, coord_y),
        otherwise the shape should be (N, 4),
        and the last dimension 4 represent
        (coord_x, coord_y, stride_w, stride_h).
    """
    feat_h, feat_w = featmap_size
    stride_w, stride_h = self.strides[level_idx]
    shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
    # keep featmap_size as Tensor instead of int, so that we
    # can convert to ONNX correctly
    shift_x = shift_x.to(dtype)

    shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
    # keep featmap_size as Tensor instead of int, so that we
    # can convert to ONNX correctly
    shift_y = shift_y.to(dtype)
    shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
    if not with_stride:
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)
    else:
        # use `feat_w * feat_h` instead of `shift_xx.shape[0]` for TensorRT
        stride_w = shift_xx.new_full((feat_w * feat_h, ), stride_w).to(dtype)
        stride_h = shift_xx.new_full((feat_w * feat_h, ), stride_h).to(dtype)
        shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
    all_points = shifts.to(device)
    return all_points
