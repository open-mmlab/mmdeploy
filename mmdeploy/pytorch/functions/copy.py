# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='copy.deepcopy')
def copy__default(tensor: Tensor, *args, **kwargs) -> Tensor:
    """Rewrite `copy.deepcopy` for default backend.

    Replace it with tensor.clone(), or may raise `NYI: Named tensors are not
    supported with the tracer`
    """
    ctx = FUNCTION_REWRITER.get_context()
    if isinstance(tensor, Tensor) and args == () and kwargs == {}:
        return tensor.clone()
    return ctx.origin_func(tensor, *args, **kwargs)
