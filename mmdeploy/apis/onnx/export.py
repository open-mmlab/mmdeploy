# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import IR, Backend, get_ir_config, get_root_logger
from .optimizer import *  # noqa
from .passes import optimize_onnx


@PIPELINE_MANAGER.register_pipeline()
def export(model: torch.nn.Module,
           args: Union[torch.Tensor, Tuple, Dict],
           output_path_prefix: str,
           backend: Union[Backend, str] = 'default',
           input_metas: Optional[Dict] = None,
           context_info: Dict = dict(),
           input_names: Optional[Sequence[str]] = None,
           output_names: Optional[Sequence[str]] = None,
           opset_version: int = 11,
           dynamic_axes: Optional[Dict] = None,
           verbose: bool = False,
           keep_initializers_as_inputs: Optional[bool] = None,
           optimize: bool = False):
    """Export a PyTorch model into ONNX format. This is a wrap of
    `torch.onnx.export` with some enhancement.

    Examples:
        >>> from mmdeploy.apis.onnx import export
        >>>
        >>> model = create_model()
        >>> args = get_input_tensor()
        >>>
        >>> export(
        >>>     model,
        >>>     args,
        >>>     'place/to/save/model',
        >>>     backend='tensorrt',
        >>>     input_names=['input'],
        >>>     output_names=['output'],
        >>>     dynamic_axes={'input': {
        >>>         0: 'batch',
        >>>         2: 'height',
        >>>         3: 'width'
        >>>     }})

    Args:
        model (torch.nn.Module): the model to be exported.
        args (torch.Tensor|Tuple|Dict): Dummy input of the model.
        output_path_prefix (str): The output file prefix. The model will
            be saved to `<output_path_prefix>.onnx`.
        backend (Backend|str): Which backend will the graph be used. Different
            backend would generate different graph.
        input_metas (Dict): The constant inputs of the model.
        context_info (Dict): The information that would be used in the context
            of exporting.
        input_names (Sequence[str]): The input names of the model.
        output_names (Sequence[str]): The output names of the model.
        opset_version (int): The version of ONNX opset version. 11 as default.
        dynamic_axes (Dict): The information used to determine which axes are
            dynamic.
        verbose (bool): Enable verbose model on `torch.onnx.export`.
        keep_initializers_as_inputs (bool): Whether we should add inputs for
            each initializer.
        optimize (bool): Perform optimize on model.
    """
    output_path = output_path_prefix + '.onnx'

    logger = get_root_logger()
    logger.info(f'Export PyTorch model to ONNX: {output_path}.')

    def _add_or_update(cfg: dict, key: str, val: Any):
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val

    context_info = deepcopy(context_info)
    deploy_cfg = context_info.pop('deploy_cfg', dict())
    ir_config = dict(
        type='onnx',
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
        keep_initializers_as_inputs=keep_initializers_as_inputs)
    _add_or_update(deploy_cfg, 'ir_config', ir_config)
    ir = IR.get(get_ir_config(deploy_cfg)['type'])
    if isinstance(backend, Backend):
        backend = backend.value
    backend_config = dict(type=backend)
    _add_or_update(deploy_cfg, 'backend_config', backend_config)

    context_info['cfg'] = deploy_cfg
    context_info['ir'] = ir
    if 'backend' not in context_info:
        context_info['backend'] = backend
    if 'opset' not in context_info:
        context_info['opset'] = opset_version

    # patch model
    patched_model = patch_model(model, cfg=deploy_cfg, backend=backend, ir=ir)

    if 'onnx_custom_passes' not in context_info:
        onnx_custom_passes = optimize_onnx if optimize else None
        context_info['onnx_custom_passes'] = onnx_custom_passes
    with RewriterContext(**context_info), torch.no_grad():
        # patch input_metas
        if input_metas is not None:
            assert isinstance(
                input_metas, dict
            ), f'Expect input_metas type is dict, get {type(input_metas)}.'
            model_forward = patched_model.forward

            def wrap_forward(forward):

                def wrapper(*arg, **kwargs):
                    return forward(*arg, **kwargs)

                return wrapper

            patched_model.forward = wrap_forward(patched_model.forward)
            patched_model.forward = partial(patched_model.forward,
                                            **input_metas)
        # force to export on cpu
        patched_model = patched_model.cpu()
        if isinstance(args, torch.Tensor):
            args = args.cpu()
        elif isinstance(args, (tuple, list)):
            args = tuple([_.cpu() for _ in args])
        else:
            raise RuntimeError(f'Not supported args: {args}')
        torch.onnx.export(
            patched_model,
            args,
            output_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            verbose=verbose)

        if input_metas is not None:
            patched_model.forward = model_forward
