# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.atan2', backend='default')
def atan2__default(
    input1: torch.Tensor,
    input2: torch.Tensor,
):
    """Rewrite `atan2` for default backend."""
    return torch.atan(input1 / (input2 + 1e-6))
