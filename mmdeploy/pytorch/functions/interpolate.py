# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from torch.autograd import Function

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend, get_root_logger


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.interpolate', backend='ncnn')
def interpolate__ncnn(ctx,
                      input: torch.Tensor,
                      size: Optional[Union[int, Tuple[int], Tuple[int, int],
                                           Tuple[int, int, int]]] = None,
                      scale_factor: Optional[Union[float,
                                                   Tuple[float]]] = None,
                      mode: str = 'nearest',
                      align_corners: Optional[bool] = None,
                      recompute_scale_factor: Optional[bool] = None):
    """Rewrite `interpolate` for ncnn backend.

    ncnn require `size` should be constant in ONNX Node. We use `scale_factor`
    instead of `size` to avoid dynamic size. To avoid rounding errors, add a
    small number when `scale_factor` is not an integer
    """

    input_size = input.shape
    if scale_factor is None:
        scale_factor = [
            s_out / s_in if int(s_out / s_in) == s_out / s_in else
            (s_out / s_in + 0.00001)
            for s_out, s_in in zip(size, input_size[2:])
        ]

    return ctx.origin_func(
        input,
        None,
        scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor)


@FUNCTION_REWRITER.register_rewriter(
    'torch.nn.functional.interpolate',
    is_pytorch=True,
    backend=Backend.TENSORRT.value)
def interpolate__tensorrt(
    ctx,
    input: torch.Tensor,
    size: Optional[Union[int, Tuple[int], Tuple[int, int], Tuple[int, int,
                                                                 int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float]]] = None,
    mode: str = 'bilinear',
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
):
    """Register default symbolic function for `interpolate`."""

    class BicubicInterpolate(Function):

        def __init__(self) -> None:
            super().__init__()

        @staticmethod
        def symbolic(g, input, scale_factor, align_corners):
            """Symbolic function for creating onnx op."""
            return g.op(
                'mmdeploy::TRTBicubicInterpolate',
                input,
                scale_factor_f=scale_factor,
                align_corners_i=align_corners)

        @staticmethod
        def forward(g, input, scale_factor, align_corners):
            """Run forward."""
            return ctx.origin_func(
                input,
                scale_factor=scale_factor,
                mode='bicubic',
                align_corners=align_corners)

    if 'bicubic' == mode:
        input_size = input.shape
        if isinstance(scale_factor, float):
            scale_factor = [scale_factor, scale_factor]
        if scale_factor is None:
            logger = get_root_logger()
            logger.warning(
                'ResizeLayer in TensorRT allow dynamic input shape with shape '
                'tensor. Which is not available for custom ops. Computed scale'
                '_factor might be the right way to get final shape.')
            scale_factor = [
                float(s_out / s_in)
                for s_out, s_in in zip(size, input_size[2:])
            ]
        return BicubicInterpolate.apply(input, scale_factor, align_corners)
    else:
        return ctx.origin_func(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor)
