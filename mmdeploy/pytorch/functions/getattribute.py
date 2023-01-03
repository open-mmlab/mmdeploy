# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.__getattribute__', backend='ncnn')
def tensor__getattribute__ncnn(self: torch.Tensor, name: str):
    """Rewrite `__getattribute__` of `torch.Tensor` for ncnn backend.

    Shape node is not supported by ncnn. This function transform dynamic shape
    to constant shape.
    """

    ctx = FUNCTION_REWRITER.get_context()
    ret = ctx.origin_func(self, name)
    if name == 'shape':
        ret = torch.Size([int(s) for s in ret])
    return ret
