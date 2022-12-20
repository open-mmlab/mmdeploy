# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import Any, Callable, Dict, Optional, Sequence

import torch

from mmdeploy.core.rewriters import FUNCTION_REWRITER
from mmdeploy.utils import IR, cfg_apply_marks, get_partition_config

MARK_FUNCTION_COUNT = dict()


def reset_mark_function_count():
    """Reset counter of mark function."""
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
    """The mark node function.

    The function does nothing but inserts a mark node to ONNX model. The mark
    can ease the process of model partition.
    """

    @staticmethod
    def symbolic(g, x, dtype, shape, func, func_id, type, name, id, attrs):
        """Symbolic function for mmdeploy::Mark op."""
        n = g.op(
            'mmdeploy::Mark',
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
    def forward(ctx, x, *args) -> torch.Tensor:
        """Run forward."""
        return x


@FUNCTION_REWRITER.register_rewriter(
    'mmdeploy.core.optimizers.function_marker.Mark.symbolic')
def mark_symbolic(rewriter, g, x, *args):
    """Rewrite symbolic of mark op."""
    if cfg_apply_marks(rewriter.cfg):
        return rewriter.origin_func(g, x, *args)
    return x


@FUNCTION_REWRITER.register_rewriter(
    'mmdeploy.core.optimizers.function_marker.Mark.forward')
def forward_of_mark(rewriter, ctx, x, dtype, shape, func, func_id, type, name,
                    id, attrs) -> torch.Tensor:
    """Rewrite forward of mark op."""
    deploy_cfg = rewriter.cfg
    # save calib data
    apply_marks = cfg_apply_marks(deploy_cfg)
    create_calib = getattr(rewriter, 'create_calib', False)
    if apply_marks and create_calib:
        partition_params = get_partition_config(deploy_cfg)
        assert partition_params is not None, 'No partition config.'
        partition_type = partition_params['type']

        from mmdeploy.apis import get_predefined_partition_cfg
        partition_cfgs = get_predefined_partition_cfg(deploy_cfg,
                                                      partition_type)
        assert hasattr(rewriter, 'calib_file')

        for partition_id, partition_cfg in enumerate(partition_cfgs):
            start = partition_cfg['start']
            if (f'{func}:{type}' not in start) and (f'{func}[{func_id}]:{type}'
                                                    not in start):
                continue

            input_name = name
            dynamic_axes = partition_cfg.get('dynamic_axes', None)
            if dynamic_axes is not None:
                input_name = name
            calib_file = rewriter.calib_file

            calib_data_group = calib_file['calib_data']
            partition_name = f'partition{partition_id}'

            if partition_name not in calib_data_group:
                calib_data_group.create_group(partition_name)
            partition_group = calib_data_group[partition_name]

            if input_name not in partition_group:
                partition_group.create_group(input_name)
            input_data_group = partition_group[input_name]

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


def mark_tensors(xs: Any, func: str, func_id: int, io_type: str, ctx: Any,
                 attrs: Dict, is_inspecting: bool, level: int) -> tuple:
    """Add mark node recursively.

    Args:
        xs (Any): Input structure which contains tensor.
        func (str): Function name of the function which xs comes from.
        func_id (int): Function index of `func` in the model.
        io_type (str): The io type of xs, `input` or `output`.
        ctx (Any): The context instance.
        attrs (Dict): The extra attributes provided by mark decorator.
        is_inspecting (bool): The names of xs are inspected or not.
        level (int): The recursive level.

    Returns:
        Any: The same structure as xs, all tensor has been replaced with Mark.
    """
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
                name = '.'.join(str(x) for x in (root, *prefix))
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


@FUNCTION_REWRITER.register_rewriter(
    'mmdeploy.core.optimizers.function_marker.mark_tensors', ir=IR.TORCHSCRIPT)
def remove_mark__torchscript(ctx, xs: Any, *args, **kwargs):
    """Disable all marks for TorchScript backend.

    As the Node `mark` is not able to be traced, we just return original input
    for the function `mark_tensors`.

    Args:
        xs (Any): Input structure which contains tensor.
    """
    return xs


def mark(func_name: Optional[str] = None,
         inputs: Optional[Sequence[str]] = None,
         outputs: Optional[Sequence[str]] = None,
         **attrs) -> Callable:
    """The decorator used to add mark node.

    Mark node can be used to support model partition.

    Args:
        func_name (str): The name of the function where marks come from.
        inputs (Sequence[str]): The input names of the marks. The final name
            might have suffix if inputs is list or dictionary.
        outputs (Sequence[str]): The output names of the marks. The final
            name might have suffix if outputs is list or dictionary.

    Returns:
        Callable: The process of mark decorator.

    Examples:
        >>> from mmdeploy.core import FUNCTION_REWRITER, mark
        >>> @FUNCTION_REWRITER.register_rewriter(
        >>>     func_name='mmdet.models.roi_heads.ConvFCBBoxHead.forward')
        >>> @mark(
        >>>     'bbox_head_forward',
        >>>     inputs=['bbox_feats'],
        >>>     outputs=['cls_score', 'bbox_pred'])
        >>> def forward_of_bbox_head(ctx, self, x):
        >>>     return ctx.origin_func(self, x)
    """
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
