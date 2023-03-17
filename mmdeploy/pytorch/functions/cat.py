# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import get_dynamic_axes


@FUNCTION_REWRITER.register_rewriter(func_name='torch.cat', backend='tensorrt')
def cat__tensorrt(tensors: Sequence[Tensor], *args, **kwargs) -> torch.Tensor:
    """Rewrite `cat` for TensorRT backend.

    cat in TensorRT does not support bool or uint8 type when input is dynamic.
    """
    ctx = FUNCTION_REWRITER.get_context()
    if get_dynamic_axes(ctx.cfg) is None:
        return ctx.origin_func(tensors, *args, **kwargs)
    if len(tensors) > 0 and (tensors[0].dtype in [torch.bool, torch.uint8]):
        original_dtype = tensors[0].dtype
        tensors = [i.to(torch.int32) for i in tensors]
        return ctx.origin_func(tensors, *args, **kwargs).to(original_dtype)
    return ctx.origin_func(tensors, *args, **kwargs)
