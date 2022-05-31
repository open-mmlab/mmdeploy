# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.size', backend='ncnn')
def tensor__size__ncnn(ctx, self, *args):
    """Rewrite `size` for ncnn backend.

    ONNX Shape node is not supported in ncnn. This function return integer
    instead of Torch.Size to avoid ONNX Shape node.
    """

    ret = ctx.origin_func(self, *args)
    if isinstance(ret, torch.Tensor):
        ret = int(ret)
    elif isinstance(ret, int):
        return (ret)
    else:
        ret = [int(r) for r in ret]
        ret = tuple(ret)
    return ret
