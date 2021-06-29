import inspect

import torch

from .function_rewriter import FUNCTION_REWRITERS


class Mark(torch.autograd.Function):

    @staticmethod
    def symbolic(g, x, type, name, id, attrs):
        n = g.op('mmcv::Mark', x, type_s=type, name_s=name, id_i=id, **attrs)
        return n

    @staticmethod
    def forward(ctx, x, *args):
        return x


@FUNCTION_REWRITERS.register_rewriter(
    'mmdeploy.utils.function_marker.Mark.symbolic')
def mark_symbolic(rewriter, g, x, *args):
    if rewriter.cfg.get('apply_marks', False):
        return rewriter.origin_func(g, x, *args)
    return x


def mark_tensors(xs, type, name, attrs):
    index = 0
    visit = set()

    def impl(ys, prefix):
        nonlocal index
        if isinstance(ys, torch.Tensor):
            if ys not in visit:
                visit.add(ys)
                index += 1
                return Mark.apply(ys, type, prefix, index - 1, attrs)
            return ys
        elif isinstance(ys, list):
            return [impl(y, f'{prefix}/{i}') for i, y in enumerate(ys)]
        elif isinstance(ys, tuple):
            return tuple(impl(y, f'{prefix}/{i}') for i, y in enumerate(ys))
        elif isinstance(ys, dict):
            return {k: impl(v, f'{prefix}/{k}') for k, v in ys.items()}
        return ys

    return impl(xs, name)


def mark(func, **attrs):
    attrs['func_s'] = func

    def decorator(f):
        params = inspect.signature(f).parameters.keys()

        def g(*args, **kwargs):
            if torch.onnx.is_in_onnx_export():
                args = [
                    mark_tensors(arg, 'input', name, attrs)
                    for name, arg in zip(params, args)
                ]
                rets = f(*args, **kwargs)
                # TODO: maybe we can traverse the AST to get the retval names?
                return mark_tensors(rets, 'output', func, attrs)
            else:
                return f(*args, **kwargs)

        return g

    return decorator
