import inspect

import torch

from mmdeploy.core.rewriters.function_rewriter import FUNCTION_REWRITER


class Mark(torch.autograd.Function):

    @staticmethod
    def symbolic(g, x, func, type, name, id, attrs):
        n = g.op(
            'mmcv::Mark',
            x,
            func_s=func,
            type_s=type,
            name_s=name,
            id_i=id,
            **attrs)
        return n

    @staticmethod
    def forward(ctx, x, *args):
        return x


@FUNCTION_REWRITER.register_rewriter(
    'mmdeploy.core.optimizers.function_marker.Mark.symbolic')
def mark_symbolic(rewriter, g, x, *args):
    if rewriter.cfg.get('apply_marks', False):
        return rewriter.origin_func(g, x, *args)
    return x


def mark_tensors(xs, func, type, ctx, attrs, is_inspecting, level):
    visit = set()
    index = 0

    def impl(ys, prefix, level):
        nonlocal index
        old_index = index
        ret = ys
        prefix = () if level == 0 else prefix

        if isinstance(ys, torch.Tensor):
            if ys not in visit:
                visit.add(ys)
                root = ctx.names[ctx.index]
                name = '/'.join(str(x) for x in (root, *prefix))
                ret = Mark.apply(ys, func, type, name, index, attrs)
                index += 1
        elif isinstance(ys, list):
            ret = [
                impl(y, prefix + (i, ), level + 1) for i, y in enumerate(ys)
            ]
        elif isinstance(ys, tuple):
            ret = tuple(
                impl(y, prefix + (i, ), level + 1) for i, y in enumerate(ys))
        elif isinstance(ys, dict):
            ret = {
                k: impl(v, prefix + (k, ), level + 1)
                for k, v in ys.items()
            }

        if level == 0 and (is_inspecting or old_index != index):
            ctx.index += 1

        return ret

    return impl(xs, (), level)


def mark(func_name=None, inputs=None, outputs=None, **attrs):

    class Context:

        def __init__(self, names):
            self.names = names
            self.index = 0

    def decorator(f):
        func = func_name if func_name else f.__name__
        is_inspect = False
        if not inputs:
            input_names = list(inspect.signature(f).parameters.keys())
            is_inspect = True
        else:
            input_names = inputs
        output_names = outputs if outputs else func

        # args and retvals match corresponding names at level 0
        args_level, rets_level = -1, -1

        if isinstance(input_names, str):
            input_names = (input_names, )

        if isinstance(output_names, str):
            output_names = (output_names, )
            rets_level += 1

        def g(*args, **kwargs):
            if torch.onnx.is_in_onnx_export():
                ctx = Context(input_names)
                args = mark_tensors(args, func, 'input', ctx, attrs,
                                    is_inspect, args_level)

                rets = f(*args, **kwargs)

                ctx = Context(output_names)
                return mark_tensors(rets, func, 'output', ctx, attrs, False,
                                    rets_level)
            else:
                return f(*args, **kwargs)

        return g

    return decorator
