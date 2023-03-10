# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch

from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import get_ir_config, get_root_logger
from mmdeploy.utils.constants import IR, Backend


def export(model: Any,
           args: Any,
           output_path: str,
           input_names: Optional[List[str]] = None,
           output_names: Optional[List[str]] = None,
           opset_version: int = 11,
           dynamic_axes: Optional[Dict] = None,
           backend: Union[Backend, str] = 'default',
           rewrite_context: Optional[Dict] = None,
           verbose: bool = False,
           const_args: Optional[Dict] = None,
           optimize: bool = True):
    """export model to ONNX.

    Examples:
        >>> from mmdeploy.ir.onnx import export
        >>>
        >>> model = create_model()
        >>> args = get_input_tensor()
        >>>
        >>> export(
        >>>     model,
        >>>     args,
        >>>     'place/to/save/model.onnx',
        >>>     backend='tensorrt',
        >>>     input_names=['input'],
        >>>     output_names=['output'],
        >>>     dynamic_axes={'input': {
        >>>         0: 'batch',
        >>>         2: 'height',
        >>>         3: 'width'
        >>>     }})

    Args:
        model (Any): Exportable PyTorch Model
        args (Any): Arguments are used to trace the graph.
        output_path (str): The output path.
        input_names (Optional[List[str]], optional): The name of the input in
            the graph. Defaults to None.
        output_names (Optional[List[str]], optional): The name of the output
            in the graph. Defaults to None.
        opset_version (int, optional): The ONNX opset version. Defaults to 11.
        dynamic_axes (Optional[Dict], optional): Dynamic axes of each inputs.
            If not given, all inputs share the fixed shapes of the args.
        verbose (bool, optional): Show detail export logs. Defaults to False.
        const_args (Optional[Dict], optional): The non-exported inputs of
            the model.
        rewrite_context (Optional[Dict], optional): The information used by
            the rewriter.
        optimize (bool): Enable optimize export model.
    """
    logger = get_root_logger()
    logger.info(f'Export PyTorch model to ONNX: {output_path}.')

    def _add_or_update(cfg: dict, key: str, val: Any):
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val

    if rewrite_context is None:
        rewrite_context = dict()

    rewrite_context = deepcopy(rewrite_context)
    # TODO: deprecate deploy_config format
    ir_config = dict(
        type='onnx',
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose)
    _add_or_update(rewrite_context, 'ir_config', ir_config)
    ir = IR.get(get_ir_config(rewrite_context)['type'])
    if isinstance(backend, Backend):
        backend = backend.value
    elif backend is None:
        backend = 'default'
    backend_config = dict(type=backend)
    _add_or_update(rewrite_context, 'backend_config', backend_config)

    # patch model
    patched_model = patch_model(
        model, cfg=rewrite_context, backend=backend, ir=ir)

    # config optimize info
    if optimize:
        from . import optimizer as _optimizer  # noqa
        if 'onnx_custom_passes' not in rewrite_context:
            from .passes import optimize_onnx
            onnx_custom_passes = optimize_onnx
        else:
            onnx_custom_passes = rewrite_context['onnx_custom_passes']
    else:
        onnx_custom_passes = None

    # start context
    with RewriterContext(
            rewrite_context,
            backend=backend,
            ir=IR.ONNX,
            opset=opset_version,
            onnx_custom_passes=onnx_custom_passes), torch.no_grad():

        if const_args is not None:
            # patch const_args
            assert isinstance(
                const_args, dict
            ), f'Expect const_args type is dict, get {type(const_args)}.'
            model_forward = patched_model.forward

            def wrap_forward(forward):

                def wrapper(*arg, **kwargs):
                    return forward(*arg, **kwargs)

                return wrapper

            patched_model.forward = wrap_forward(patched_model.forward)
            patched_model.forward = partial(patched_model.forward,
                                            **const_args)

        # export with torch.onnx.export
        torch.onnx.export(
            patched_model,
            args,
            output_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=False,
            verbose=verbose)

        if const_args is not None:
            # recovery forward
            patched_model.forward = model_forward
