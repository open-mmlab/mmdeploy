import inspect

import torch

from mmdeploy.core.rewriters.function_rewriter import FUNCTION_REWRITER

MARK_FUNCTION_COUNT = dict()


def reset_mark_function_count():
    for k in MARK_FUNCTION_COUNT:
        MARK_FUNCTION_COUNT[k] = 0


TORCH_DTYPE_TO_ONNX = {
    torch.uint8: torch.onnx.TensorProtoDataType.UINT8,
    torch.int8: torch.onnx.TensorProtoDataType.INT8,
    torch.float64: torch.onnx.TensorProtoDataType.DOUBLE,
    torch.float32: torch.onnx.TensorProtoDataType.FLOAT,
    torch.float16: torch.onnx.TensorProtoDataType.FLOAT16,
    torch.int32: torch.onnx.TensorProtoDataType.INT32,
    torch.int64: torch.onnx.TensorProtoDataType.INT64,
    torch.int16: torch.onnx.TensorProtoDataType.INT16,
    torch.bool: torch.onnx.TensorProtoDataType.BOOL,
    torch.complex64: torch.onnx.TensorProtoDataType.COMPLEX64,
    torch.complex128: torch.onnx.TensorProtoDataType.COMPLEX128,
}


class Mark(torch.autograd.Function):

    @staticmethod
    def symbolic(g, x, dtype, shape, func, func_id, type, name, id, attrs):
        n = g.op(
            'mmcv::Mark',
            x,
            dtype_i=TORCH_DTYPE_TO_ONNX[dtype],
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


@FUNCTION_REWRITER.register_rewriter(
    'mmdeploy.core.optimizers.function_marker.Mark.forward')
def forward_of_mark(rewriter, ctx, x, dtype, shape, func, func_id, type, name,
                    id, attrs):
    deploy_cfg = rewriter.cfg
    # save calib data
    apply_marks = deploy_cfg.get('apply_marks', False)
    create_calib = getattr(rewriter, 'create_calib', False)
    if apply_marks and create_calib:
        codebase = deploy_cfg['codebase']
        assert 'split_params' in deploy_cfg
        split_params = deploy_cfg['split_params']
        split_type = split_params['split_type']
        from mmdeploy.apis.utils import get_split_cfg
        split_cfgs = get_split_cfg(codebase, split_type)
        assert hasattr(rewriter, 'calib_file')

        for split_id, split_cfg in enumerate(split_cfgs):
            start = split_cfg['start']
            if (f'{func}:{type}' not in start) and (f'{func}[{func_id}]:{type}'
                                                    not in start):
                continue

            input_name = name
            dynamic_axes = split_cfg.get('dynamic_axes', None)
            if dynamic_axes is not None:
                input_name = name
            calib_file = rewriter.calib_file

            calib_data_group = calib_file['calib_data']
            split_name = f'split{split_id}'

            if split_name not in calib_data_group:
                calib_data_group.create_group(split_name)
            split_group = calib_data_group[split_name]

            if input_name not in split_group:
                split_group.create_group(input_name)
            input_data_group = split_group[input_name]

            data_id = rewriter.data_id
            x_np = x.detach().cpu().numpy()
            input_data_group.create_dataset(
                str(data_id),
                shape=x_np.shape,
                compression='gzip',
                compression_opts=4,
                data=x_np)

    return rewriter.origin_func(ctx, x, dtype, shape, func, func_id, type,
                                name, id, attrs)


def mark_tensors(xs, func, func_id, io_type, ctx, attrs, is_inspecting, level):
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
                ret = Mark.apply(ys, ys.dtype, ys_shape, func, func_id,
                                 io_type, name, index, attrs)
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
