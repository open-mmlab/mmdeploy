# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import IR, Backend, get_ir_config, get_root_logger


def trace(func: torch.nn.Module,
          inputs: Union[torch.Tensor, Tuple],
          output_path: Optional[str] = None,
          backend: Union[Backend, str] = 'default',
          rewrite_context: Dict = dict(),
          check_trace: bool = True,
          check_tolerance: float = 1e-05,
          const_args: Optional[Dict] = None) -> torch.jit.TracedModule:
    """A wrapper of `torch.jit.trace` with some enhancement.

    Examples:
        >>> from mmdeploy.ir.torchscript import export
        >>>
        >>> func = create_model()
        >>> inputs = get_input_tensor()
        >>>
        >>> jit_model = export(
        >>>     func,
        >>>     inputs,
        >>>     output_path,
        >>>     backend='torchscript',
        >>>     check_trace=False)
        >>>

    Args:
        func (torch.nn.Module): A Python function or `torch.nn.Module` that
            will be run with `example_inputs`.
        inputs (torch.Tensor, Tuple): A tuple of example inputs that will be
            passed to the function while tracing.
        output_path (str): The output path.
        backend (Backend|str): Which backend will the graph be used. Different
            backend would generate different graph.
        const_args (Dict): The constant inputs of the model.
        rewrite_context (Dict): The information that would be used in the
            context of exporting.
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

    if rewrite_context is None:
        rewrite_context = dict()

    rewrite_context = deepcopy(rewrite_context)
    ir_config = dict(type='torchscript')
    _add_or_update(rewrite_context, 'ir_config', ir_config)

    if isinstance(backend, Backend):
        backend = backend.value
    elif backend is None:
        backend = 'default'
    backend_config = dict(type=backend)
    _add_or_update(rewrite_context, 'backend_config', backend_config)

    # patch model
    if isinstance(func, torch.nn.Module):
        ir = IR.get(get_ir_config(rewrite_context)['type'])
        func = patch_model(func, cfg=rewrite_context, backend=backend, ir=ir)

    with RewriterContext(
            rewrite_context, ir=IR.TORCHSCRIPT,
            backend=backend), torch.no_grad():

        # patch const_args
        if const_args is not None:
            assert isinstance(
                const_args, dict
            ), f'Expect const_args type is dict, get {type(const_args)}.'
            model_forward = func.forward
            func.forward = partial(func.forward, **const_args)

        # for exporting models with weight that depends on inputs
        func(*inputs) if isinstance(inputs, Sequence) \
            else func(inputs)
        ts_model = torch.jit.trace(
            func,
            inputs,
            check_trace=check_trace,
            check_tolerance=check_tolerance)

        if const_args is not None:
            func.forward = model_forward

    # save model
    logger.info(f'Save PyTorch model: {output_path}.')
    torch.jit.save(ts_model, output_path)

    return ts_model
