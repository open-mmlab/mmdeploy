import torch

from mmdeploy.utils import FUNCTION_REWRITERS


@FUNCTION_REWRITERS.register_rewriter(func_name='torch.topk', backend='default')
@FUNCTION_REWRITERS.register_rewriter(func_name='torch.Tensor.topk', backend='default')
def rewrite_topk_default(rewriter, input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k, device=input.device, dtype=torch.long)
    # Always keep topk op for dynamic input
    k = torch.where(k < size, k, size)
    return rewriter.origin_func(input, k, dim=dim, largest=largest, sorted=sorted)


@FUNCTION_REWRITERS.register_rewriter(func_name='torch.topk', backend='tensorrt')
@FUNCTION_REWRITERS.register_rewriter(func_name='torch.Tensor.topk', backend='tensorrt')
def rewrite_topk_tensorrt(rewriter, input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if k > size:
        k = size
    if not isinstance(k, int):
        k = int(k)
    return rewriter.origin_func(input, k, dim=dim, largest=largest, sorted=sorted)
