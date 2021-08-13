import inspect

import torch
from torch.onnx.symbolic_helper import cast_pytorch_to_onnx

from mmdeploy.core.rewriters.function_rewriter import FUNCTION_REWRITER

MARK_FUNCTION_COUNT = dict()


def reset_mark_function_count():
    for k in MARK_FUNCTION_COUNT:
        MARK_FUNCTION_COUNT[k] = 0


class Mark(torch.autograd.Function):

    @staticmethod
    def symbolic(g, x, shape, func, func_id, type, name, id, attrs):
        n = g.op(
            'mmcv::Mark',
            x,
            dtype_i=cast_pytorch_to_onnx[x.type().scalarType()].value,
            shape_i=shape,
            func_s=func,
            func_id_i=func_id,
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


def mark_tensors(xs, func, func_id, type, ctx, attrs, is_inspecting, level):
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
                ys_shape = tuple(int(s) for s in ys.shape)
                ret = Mark.apply(ys, ys_shape, func, func_id, type, name,
                                 index, attrs)
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
    MARK_FUNCTION_COUNT[func_name] = 0

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
            func_id = MARK_FUNCTION_COUNT[func_name]
            MARK_FUNCTION_COUNT[func_name] += 1
            ctx = Context(input_names)
            args = mark_tensors(args, func, func_id, 'input', ctx, attrs,
                                is_inspect, args_level)

            rets = f(*args, **kwargs)

            ctx = Context(output_names)
            func_ret = mark_tensors(rets, func, func_id, 'output', ctx, attrs,
                                    False, rets_level)
            return func_ret

        return g

    return decorator
