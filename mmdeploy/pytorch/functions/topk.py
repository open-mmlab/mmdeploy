import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(func_name='torch.topk', backend='default')
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.topk', backend='default')
def topk_dynamic(ctx, input, k, dim=None, largest=True, sorted=True):
    """Rewrite `topk` for default backend."""

    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k, device=input.device, dtype=torch.long)
    # Always keep topk op for dynamic input
    if isinstance(size, torch.Tensor):
        size = size.to(input.device)
    k = torch.where(k < size, k, size)
    return ctx.origin_func(input, k, dim=dim, largest=largest, sorted=sorted)


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.topk', backend='tensorrt')
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.topk', backend='tensorrt')
def topk_static(ctx, input, k, dim=None, largest=True, sorted=True):
    """Rewrite `topk` for TensorRT backend."""

    if dim is None:
        dim = int(input.ndim - 1)
    size = input.shape[dim]
    if k > size:
        k = size
    if not isinstance(k, int):
        k = int(k)
    return ctx.origin_func(input, k, dim=dim, largest=largest, sorted=sorted)
