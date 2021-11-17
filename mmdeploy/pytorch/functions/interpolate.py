from typing import Optional, Tuple, Union

import torch

from mmdeploy.core import FUNCTION_REWRITER


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
    """Rewrite `interpolate` for NCNN backend.

    NCNN require `size` should be constant in ONNX Node. We use `scale_factor`
    instead of `size` to avoid dynamic size.
    """

    input_size = input.shape
    if scale_factor is None:
        scale_factor = [
            s_out / s_in for s_out, s_in in zip(size, input_size[2:])
        ]

    return ctx.origin_func(
        input,
        None,
        scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor)
