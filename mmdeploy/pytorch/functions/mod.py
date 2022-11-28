# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from packaging import version

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils.constants import Backend


# TODO add version control when MOD is supported by TensorRT
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.__mod__', backend=Backend.TENSORRT.value)
def mod__tensorrt(ctx, input: torch.Tensor, other: Union[torch.Tensor,
                                                         torch.NumberType],
                  *args, **kwargs) -> torch.Tensor:
    """Rewrite `mod` when exporting model to ONNX for TensorRT backend."""
    if version.parse(torch.__version__) > version.parse('1.10.0'):
        return input - (input // other) * other
    return ctx.origin_func(input, other, *args, **kwargs)
