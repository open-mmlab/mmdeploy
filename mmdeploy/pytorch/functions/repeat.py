# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.repeat', backend='tensorrt')
def tensor__repeat__tensorrt(ctx, input: torch.Tensor,
                             *size: Union[torch.Size, Sequence[int]]):
    """Rewrite `repeat` for TensorRT backend.

    Some layers in TensorRT can not be applied on batch axis. add extra axis
    before operation and remove it afterward.
    """

    origin_func = ctx.origin_func
    if input.dim() == 1 and len(size) == 1:
        return origin_func(input.unsqueeze(0), *([1] + list(size))).squeeze(0)
    else:
        return origin_func(input, *size)
