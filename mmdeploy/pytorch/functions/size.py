import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.size', backend='ncnn')
def size_of_tensor_static(ctx, self, *args):
    """Rewrite `size` for NCNN backend."""

    ret = ctx.origin_func(self, *args)
    if isinstance(ret, torch.Tensor):
        ret = int(ret)
    else:
        ret = [int(r) for r in ret]
        ret = tuple(ret)
    return ret
