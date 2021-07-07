import inspect

import torch

from .function_rewriter import FUNCTION_REWRITERS


class Mark(torch.autograd.Function):

    @staticmethod
    def symbolic(g, x, func, type, name, id, attrs):
        n = g.op("mmcv::Mark", x, func_s=func, type_s=type,
                 name_s=name, id_i=id, **attrs)
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


def mark_tensors(xs, func, type, index, name, attrs):
    visit = set()

    def impl(ys, prefix):
        nonlocal index
        if isinstance(ys, torch.Tensor):
            if ys not in visit:
                visit.add(ys)
                index += 1
                return Mark.apply(ys, func, type, prefix, index - 1, attrs)
            return ys
        elif isinstance(ys, list):
            return [impl(y, f'{prefix}/{i}') for i, y in enumerate(ys)]
        elif isinstance(ys, tuple):
            return tuple(impl(y, f'{prefix}/{i}') for i, y in enumerate(ys))
        elif isinstance(ys, dict):
            return {k: impl(v, f'{prefix}/{k}') for k, v in ys.items()}
        return ys

    return impl(xs, name)


def handle_extra_params(params, args):
    n_params = len(params) if isinstance(params, list) else 1
    n_pad = len(args) - n_params
    assert n_pad >= 0
    if n_pad:
        if not isinstance(params, list):
            params = [params]
        return ['_'] * n_pad + params
    else:
        return params


def mark(func_name=None, inputs=None, outputs=None, **attrs):
    if isinstance(inputs, tuple):
        inputs = list(inputs)

    def decorator(f):
        func = func_name if func_name else f.__name__
        # if no input name specified, fallback to function signature
        param_names = inputs if inputs else list(
            inspect.signature(f).parameters.keys())
        # simply use func as fallback name since inspecting return statement
        # is non-trivial
        # TODO: maybe we can traverse the AST of f to get the retval names?
        output_names = outputs if outputs else func

        def g(*args, **kwargs):
            if torch.onnx.is_in_onnx_export():
                # pad param_names to avoid 'rewriter' and 'self'
                arg_names = handle_extra_params(param_names, args)
                if isinstance(arg_names, (list, tuple)):
                    arg_type = type(args)
                    args = arg_type(
                        mark_tensors(arg, func, 'input', i, name, attrs)
                        for i, (name, arg) in enumerate(zip(arg_names, args)))
                else:
                    args = mark_tensors(
                        args, func, 'input', 0, arg_names, attrs)

                rets = f(*args, **kwargs)

                if isinstance(output_names, (list, tuple)):
                    ret_type = type(rets)
                    return ret_type(
                        mark_tensors(ret, func, 'output', i, name, attrs)
                        for i, (name, ret) in enumerate(zip(output_names, rets)))
                else:
                    return mark_tensors(
                        rets, func, 'output', 0, output_names, attrs)
            else:
                return f(*args, **kwargs)

        return g

    return decorator
