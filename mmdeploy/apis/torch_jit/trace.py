# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import IR, Backend, get_ir_config, get_root_logger
from ..core import PIPELINE_MANAGER


@PIPELINE_MANAGER.register_pipeline()
def trace(func: torch.nn.Module,
          inputs: Union[torch.Tensor, Tuple],
          output_path_prefix: Optional[str] = None,
          backend: Union[Backend, str] = 'default',
          context_info: Dict = dict(),
          check_trace: bool = True,
          check_tolerance: float = 1e-05) -> torch.jit.TracedModule:
    """A wrapper of `torch.jit.trace` with some enhancement.

    Examples:
        >>> from mmdeploy.apis.torch_jit import trace
        >>>
        >>> func = create_model()
        >>> inputs = get_input_tensor()
        >>>
        >>> jit_model = trace(
        >>>     func,
        >>>     inputs,
        >>>     backend='torchscript',
        >>>     check_trace=False)
        >>>

    Args:
        func (torch.nn.Module): A Python function or `torch.nn.Module` that
            will be run with `example_inputs`.
        inputs (torch.Tensor, Tuple): A tuple of example inputs that will be
            passed to the function while tracing.
        output_path_prefix (str): The model would be serialized in
            `<output_path_prefix>.pth`, None if you don't want to
            save the model.
        backend (Backend|str): Which backend will the graph be used. Different
            backend would generate different graph.
        context_info (Dict): The information that would be used in the context
            of exporting.
        check_trace (bool): Check if the same inputs run through traced code
            produce the same outputs.
        check_tolerance (float): Floating-point comparison tolerance to use in
            the checker procedure.

    Returns:
        torch.jit.TracedModule: The traced torch jit model.
    """
    logger = get_root_logger()
    logger.info('Export PyTorch model to torchscript.')

    def _add_or_update(cfg: dict, key: str, val: Any):
        if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val

    context_info = deepcopy(context_info)
    deploy_cfg = context_info.pop('deploy_cfg', dict())
    ir_config = dict(type='torchscript')
    _add_or_update(deploy_cfg, 'ir_config', ir_config)

    if isinstance(backend, Backend):
        backend = backend.value
    backend_config = dict(type=backend)
    _add_or_update(deploy_cfg, 'backend_config', backend_config)

    context_info['cfg'] = deploy_cfg
    if 'backend' not in context_info:
        context_info['backend'] = backend
    elif context_info['backend'] != backend:
        logger.warning(
            f'Find backend {context_info["backend"]} in context_info.'
            f' Expect {backend}.')
    if 'ir' not in context_info:
        context_info['ir'] = IR.TORCHSCRIPT
    elif context_info['ir'] != backend:
        logger.warning(f'Find ir {context_info["ir"]} in context_info.'
                       f' Expect {IR.TORCHSCRIPT}.')

    # patch model
    if isinstance(func, torch.nn.Module):
        ir = IR.get(get_ir_config(deploy_cfg)['type'])
        func = patch_model(func, cfg=deploy_cfg, backend=backend, ir=ir)

    with RewriterContext(**context_info), torch.no_grad():
        # for exporting models with weight that depends on inputs
        func(*inputs) if isinstance(inputs, Sequence) \
            else func(inputs)
        ts_model = torch.jit.trace(
            func,
            inputs,
            check_trace=check_trace,
            check_tolerance=check_tolerance)

    # save model
    if output_path_prefix is not None:
        output_path = output_path_prefix + '.pt'
        logger.info(f'Save PyTorch model: {output_path}.')
        torch.jit.save(ts_model, output_path)

    return ts_model
