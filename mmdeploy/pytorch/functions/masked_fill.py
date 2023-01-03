# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from torch.types import Number

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils.constants import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.masked_fill', backend=Backend.ONNXRUNTIME.value)
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.masked_fill', backend=Backend.ONNXRUNTIME.value)
def masked_fill__onnxruntime(
        input, mask: torch.Tensor, value: Union[torch.Tensor,
                                                Number]) -> torch.Tensor:
    """Rewrite `masked_fill` for onnxruntime backend.

    SATRN model as example, when value is set to `float('-inf')`, the results
    of ORT inferencing turns out to be NAN.
    """
    ctx = FUNCTION_REWRITER.get_context()
    if value == float('-inf'):
        value = -1e34  # hard coding number
    return ctx.origin_func(input, mask, value)
