# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch

from mmdeploy.ir.torchscript import export
from mmdeploy.utils import Backend
from ..core import PIPELINE_MANAGER


@PIPELINE_MANAGER.register_pipeline()
def trace(func: torch.nn.Module,
          inputs: Union[torch.Tensor, Tuple],
          output_path_prefix: Optional[str] = None,
          backend: Union[Backend, str] = 'default',
          input_metas: Optional[Dict] = None,
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
        >>>  trace(
        >>>     func,
        >>>     inputs,
        >>>     output_prefix,
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
        input_metas (Dict): The constant inputs of the model.
        context_info (Dict): The information that would be used in the context
            of exporting.
        check_trace (bool): Check if the same inputs run through traced code
            produce the same outputs.
        check_tolerance (float): Floating-point comparison tolerance to use in
            the checker procedure.

    Returns:
        torch.jit.TracedModule: The traced torch jit model.
    """
    if output_path_prefix is None:
        from tempfile import NamedTemporaryFile
        output_path = NamedTemporaryFile(suffix='.pth').name
    else:
        output_path = output_path_prefix + '.pth'

    deploy_cfg = context_info.pop('deploy_cfg', dict())
    export(
        func,
        inputs,
        output_path,
        backend=backend,
        rewrite_context=deploy_cfg,
        check_trace=check_trace,
        check_tolerance=check_tolerance,
        const_args=input_metas)

    ts_model = torch.jit.load(output_path)

    return ts_model
