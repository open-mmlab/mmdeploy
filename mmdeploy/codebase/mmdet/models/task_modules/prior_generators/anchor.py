# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch.onnx import symbolic_helper

from mmdeploy.core import FUNCTION_REWRITER


class GridPriorsTRTOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, base_anchors, feat_h, feat_w, stride_h: int,
                stride_w: int):
        device = base_anchors.device
        dtype = base_anchors.dtype
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        def _meshgrid(x, y, row_major=True):
            # use shape instead of len to keep tracing while exporting to onnx
            xx = x.repeat(y.shape[0])
            yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
            if row_major:
                return xx, yy
            else:
                return yy, xx

        shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # then (0, 1), (0, 2), ...
        return all_anchors

    @staticmethod
    @symbolic_helper.parse_args('v', 'v', 'v', 'i', 'i')
    def symbolic(g, base_anchors, feat_h, feat_w, stride_h: int,
                 stride_w: int):
        # zero_h and zero_w is used to provide shape to GridPriorsTRT
        feat_h = symbolic_helper._unsqueeze_helper(g, feat_h, [0])
        feat_w = symbolic_helper._unsqueeze_helper(g, feat_w, [0])
        zero_h = g.op(
            'ConstantOfShape',
            feat_h,
            value_t=torch.tensor([0], dtype=torch.long),
        )
        zero_w = g.op(
            'ConstantOfShape',
            feat_w,
            value_t=torch.tensor([0], dtype=torch.long),
        )
        return g.op(
            'mmdeploy::GridPriorsTRT',
            base_anchors,
            zero_h,
            zero_w,
            stride_h_i=stride_h,
            stride_w_i=stride_w)


grid_priors_trt = GridPriorsTRTOp.apply


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.core.anchor.anchor_generator.'
    'AnchorGenerator.single_level_grid_priors',
    backend='tensorrt')
def anchorgenerator__single_level_grid_priors__trt(
        ctx,
        self,
        featmap_size: Tuple[int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda') -> torch.Tensor:
    """This is a rewrite to replace ONNX anchor generator to TensorRT custom
    op.

    Args:
        ctx : The rewriter context
        featmap_size (tuple[int]): Size of the feature maps.
        level_idx (int): The index of corresponding feature map level.
        dtype (obj:`torch.dtype`): Date type of points.Defaults to
            ``torch.float32``.
        device (str, optional): The device the tensor will be put on.
            Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
    """
    feat_h, feat_w = featmap_size
    if isinstance(feat_h, int) and isinstance(feat_w, int):
        return ctx.origin_func(self, featmap_size, level_idx, dtype,
                               device).data
    base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
    stride_w, stride_h = self.strides[level_idx]
    return grid_priors_trt(base_anchors, feat_h, feat_w, stride_h, stride_w)
