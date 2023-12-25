# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import torch

from mmdeploy.core import FUNCTION_REWRITER

@FUNCTION_REWRITER.register_rewriter(func_name='torch.distributed')
def distributed_rewriter():
    """rewrite torch.distributed to support some embedding device for higher PyTorch"""
    # check torch.distributed is available?
    if not torch.distributed.is_available():
        torch.distributed.ReduceOp = lambda: None
