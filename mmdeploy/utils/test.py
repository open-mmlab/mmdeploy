# Copyright (c) OpenMMLab. All rights reserved.

import copy
import os.path as osp
import random
import string
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import torch
from mmengine import Config
from mmengine.model import BaseModel
from torch import nn

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close

import mmdeploy.codebase  # noqa: F401,F403
from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import (IR, Backend, get_backend, get_dynamic_axes,
                            get_ir_config, get_onnx_config)


def backend_checker(backend: Backend, require_plugin: bool = False):
    """A decorator which checks if a backend is available.

    Args:
        backend (Backend): The backend needs to be checked.
        require_plugin (bool): The checker will check if the backend package
            has been installed. If this variable is `True`, then the checker
            will also check if the backend plugin has been compiled. Default
            to `False`.
    """
    from mmdeploy.backend.base import get_backend_manager

    backend_mgr = get_backend_manager(backend.value)
    result = backend_mgr.is_available(with_custom_ops=require_plugin)

    checker = pytest.mark.skipif(
        not result, reason=f'{backend.value} package is not available')

    return checker


def check_backend(backend: Backend, require_plugin: bool = False):
    """A utility to check if a backend is available.

    Args:
        backend (Backend): The backend needs to be checked.
        require_plugin (bool): The checker will check if the backend package
            has been installed. If this variable is `True`, then the checker
            will also check if the backend plugin has been compiled. Default
            to `False`.
    """
    from mmdeploy.backend.base import get_backend_manager

    backend_mgr = get_backend_manager(backend.value)
    result = backend_mgr.is_available(with_custom_ops=require_plugin)

    if not result:
        pytest.skip(f'{backend.value} package is not available')


class WrapFunction(nn.Module):
    """Wrap a pytorch function to nn.Module.

    It serves as partial function and can be exportable to ONNX.

    Args:
        wrapped_function (Callable): Input function to be wrapped.

    Examples:
        >>> from mmdeploy.utils.test import WrapFunction
        >>> import torch
        >>>
        >>> def clip(x, min, max):
        >>>     return torch.clamp(x, min, max)
        >>>
        >>> wrapped_model = WrapFunction(clip)
    """

    def __init__(self, wrapped_function: Callable, **kwargs):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function
        self.kwargs = kwargs

    def forward(self, *args, **kwargs) -> Any:
        """Call the wrapped function."""
        kwargs.update(self.kwargs)
        return self.wrapped_function(*args, **kwargs)


class WrapModel(nn.Module):
    """A wrapper class for rewrite unittests.

    It serves as partial function but can be rewritten with RewriterContext.

    Args:
        model (nn.Module): A pytorch module.
        func_name (str): Which function to use as forward function.

    Examples:
        >>> from mmdeploy.utils.test import WrapModel
        >>> from mmdet.models import AnchorHead
        >>>
        >>> model = AnchorHead(num_classes=4, in_channels=1)
        >>> wrapped_model = WrapModel(anchor_head,
        >>>                           'get_bboxes',
        >>>                           with_nms=True)
    """

    def __init__(self, model: nn.Module, func_name: str, **kwargs):
        super(WrapModel, self).__init__()
        assert hasattr(model,
                       func_name), f'Got unexpected func name: {func_name}'
        self.model = model
        self.kwargs = kwargs
        self.func_name = func_name

    def forward(self, *args, **kwargs):
        """Run forward of the model."""
        kwargs.update(self.kwargs)
        func = getattr(self.model, self.func_name)
        return func(*args, **kwargs)


class DummyModel(BaseModel):
    """A dummy model for unit tests.

    Args:
        outputs (Any): Predefined output variables.
    """

    def __init__(self, outputs=None, *args, **kwargs):
        super().__init__()
        self.outputs = outputs

    def forward(self, *args, **kwargs):
        """Run forward."""
        return self.outputs


class SwitchBackendWrapper:
    """A switcher for backend wrapper for unit tests.
    Examples:
        >>> from mmdeploy.utils.test import SwitchBackendWrapper
        >>> from mmdeploy.backend.onnxruntime import ORTWrapper
        >>> with SwitchBackendWrapper(ORTWrapper) as wrapper:
        >>>     wrapper.set(ORTWrapper, outputs=outputs)
        >>>     ...
        >>>     # ORTWrapper will recover when exiting context
        >>> ...
    """
    init = None
    forward = None
    call = None

    class BackendWrapper(torch.nn.Module):
        """A dummy backend wrapper for unit tests.

        To enable BaseWrapper.output_to_list(), the wrapper needs member
        variable `_output_names` that is set in constructor. Therefore,
        the dummy BackendWrapper needs a constructor that receives
        output_names.

        Args:
            output_names (Any): `output_name` of BaseWrapper
        """

        def __init__(self, output_names=['dets', 'labels'], *args, **kwargs):
            torch.nn.Module.__init__(self)
            self._output_names = output_names

        def forward(self, *args, **kwargs):
            """Run forward."""
            return self.outputs

        def __call__(self, *args, **kwds):
            """Call the forward method."""
            return self.forward(*args, **kwds)

    def __init__(self, recover_class):
        self._recover_class = recover_class

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        self.recover()

    def set(self, **kwargs):
        """Replace attributes in backend wrappers with dummy items."""
        obj = self._recover_class
        self.init = obj.__init__
        self.forward = obj.forward
        self.call = obj.__call__
        obj.__init__ = SwitchBackendWrapper.BackendWrapper.__init__
        obj.forward = SwitchBackendWrapper.BackendWrapper.forward
        obj.__call__ = SwitchBackendWrapper.BackendWrapper.__call__
        for k, v in kwargs.items():
            setattr(obj, k, v)

    def recover(self):
        """Recover to original class."""
        assert self.init is not None and \
            self.forward is not None,\
            'recover method must be called after exchange'
        obj = self._recover_class
        obj.__init__ = self.init
        obj.forward = self.forward
        obj.__call__ = self.call


def assert_allclose(expected: List[Union[torch.Tensor, np.ndarray]],
                    actual: List[Union[torch.Tensor, np.ndarray]],
                    tolerate_small_mismatch: bool = False):
    """Determine whether all actual values are closed with the expected values.

    Args:
        actual (list[torch.Tensor | np.ndarray): Actual value.
        expected (list[torch.Tensor | np.ndarray): Expected value.
        tolerate_small_mismatch (bool): Whether tolerate small mismatch,
        Default is False.
    """
    if not (isinstance(expected, list) and isinstance(actual, list)):
        raise ValueError('Argument desired and actual should be a list')
    if len(expected) != len(actual):
        raise ValueError('Length of desired and actual should be equal')

    for i in range(0, len(expected)):
        if isinstance(expected[i], (list, np.ndarray)):
            expected[i] = torch.tensor(expected[i])
        if isinstance(actual[i], (list, np.ndarray)):
            actual[i] = torch.tensor(actual[i])
        try:
            torch_assert_close(actual[i], expected[i], rtol=1e-03, atol=1e-05)
        except AssertionError as error:
            if tolerate_small_mismatch:
                assert '(0.00%)' in str(error), str(error)
            else:
                raise


def get_model_outputs(model: nn.Module, func_name: str,
                      model_inputs: dict) -> Any:
    """To get outputs of pytorch model.

    Args:
        model (nn.Module): The input model.
        func_name (str): The function of model.
        model_inputs (dict): The inputs for model.

    Returns:
        Any: The output of model, decided by the input model.
    """
    assert hasattr(model, func_name), f'Got unexpected func name: {func_name}'
    func = getattr(model, func_name)
    model_outputs = func(**copy.deepcopy(model_inputs))
    return model_outputs


def get_flatten_inputs(
    model_inputs: Dict[str, Union[Tuple, List, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """This function unwraps lists and tuples from 'model_inputs' and assigns a
    unique name to their values.

    Example:
        model_inputs = {'A': list(tensor_0, tensor_1), 'B': tensor_3}
        flatten_inputs = get_flatten_inputs(model_inputs)
        flatten_inputs: {'A_0': tensor_0, 'A_1': tensor_1, 'B': tensor_3}

    Args:
        model_inputs (dict): Key-value pairs of model inputs with
            lists and tuples.

    Returns:
        Dict[str, torch.Tensor]: Key-value pairs of model inputs with
            unwrapped lists and tuples.
    """
    flatten_inputs = {}
    for name, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            flatten_inputs[name] = value
        elif isinstance(value, (list, tuple)):
            if len(value) == 1:
                flatten_inputs[name] = value[0]
            else:
                for i, tensor in enumerate(value):
                    name_i = f'{name}_{i}'
                    flatten_inputs[name_i] = tensor
    return flatten_inputs


def get_onnx_model(wrapped_model: nn.Module,
                   model_inputs: Dict[str, Union[Tuple, List, torch.Tensor]],
                   deploy_cfg: Config) -> str:
    """To get path to onnx model after export.

    Args:
        wrapped_model (nn.Module): The input model.
        model_inputs (dict): Inputs for model.
        deploy_cfg (Config): Deployment config.

    Returns:
        str: The path to the ONNX model file.
    """
    onnx_file_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
    onnx_cfg = get_onnx_config(deploy_cfg)
    backend = get_backend(deploy_cfg)
    patched_model = patch_model(
        wrapped_model, cfg=deploy_cfg, backend=backend.value)
    flatten_model_inputs = get_flatten_inputs(model_inputs)
    input_names = onnx_cfg.get('input_names', None)
    if input_names is None:
        input_names = [
            k for k, v in flatten_model_inputs.items() if k != 'ctx'
        ]
    output_names = onnx_cfg.get('output_names', None)
    dynamic_axes = get_dynamic_axes(deploy_cfg, input_names)

    class DummyModel(torch.nn.Module):

        def __init__(self):
            super(DummyModel, self).__init__()
            self.model = patched_model

        def forward(self, inputs: dict):
            return self.model(**inputs)

    model = DummyModel().eval()

    with RewriterContext(
            cfg=deploy_cfg, backend=backend.value, opset=11), torch.no_grad():
        torch.onnx.export(
            model, (model_inputs, {}),
            onnx_file_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=11,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=False)
    return onnx_file_path


def get_ts_model(wrapped_model: nn.Module,
                 model_inputs: Dict[str, Union[Tuple, List, torch.Tensor]],
                 deploy_cfg: Config) -> str:
    """To get path to onnx model after export.

    Args:
        wrapped_model (nn.Module): The input model.
        model_inputs (dict): Inputs for model.
        deploy_cfg (Config): Deployment config.

    Returns:
        str: The path to the TorchScript model file.
    """
    ir_file_path = tempfile.NamedTemporaryFile(suffix='.pt').name
    backend = get_backend(deploy_cfg)

    from mmdeploy.apis.torch_jit import trace
    context_info = dict(deploy_cfg=deploy_cfg)
    output_prefix = osp.splitext(ir_file_path)[0]

    example_inputs = [v for _, v in model_inputs.items()]
    trace(
        wrapped_model,
        example_inputs,
        output_path_prefix=output_prefix,
        backend=backend,
        context_info=context_info)
    return ir_file_path


def get_backend_outputs(ir_file_path: str,
                        model_inputs: Dict[str, Union[Tuple, List,
                                                      torch.Tensor]],
                        deploy_cfg: Config) -> Union[Any, None]:
    """To get backend outputs of model.

    Args:
        ir_file_path (str): The path to the IR file.
        model_inputs (dict): Inputs for model.
        deploy_cfg (Config): Deployment config.

    Returns:
        Union[Any, None]: The outputs of model, decided by the backend wrapper.
            If the backend specified in 'deploy_cfg' is not available,
            then None will be returned.
    """
    from mmdeploy.apis.utils import to_backend
    backend = get_backend(deploy_cfg)
    flatten_model_inputs = get_flatten_inputs(model_inputs)
    ir_config = get_ir_config(deploy_cfg)
    input_names = ir_config.get('input_names', None)
    output_names = ir_config.get('output_names', None)
    if input_names is None:
        input_names = [
            k for k, v in flatten_model_inputs.items() if k != 'ctx'
        ]

    work_dir = tempfile.TemporaryDirectory().name
    device = 'cpu'

    # TODO: Try to wrap these code into backend manager
    if backend != Backend.TORCHSCRIPT:
        model_inputs = flatten_model_inputs
    if backend == Backend.TENSORRT:
        device = 'cuda'
        model_inputs = dict((k, v.cuda()) for k, v in model_inputs.items())
    elif backend == Backend.OPENVINO:
        input_info = {
            name: value.shape
            for name, value in flatten_model_inputs.items()
        }
        deploy_cfg['backend_config']['model_inputs'] = [
            dict(opt_shapes=input_info)
        ]
    backend_files = to_backend(
        backend.value, [ir_file_path],
        work_dir=work_dir,
        deploy_cfg=deploy_cfg,
        device=device)
    backend_feats = model_inputs

    if backend == Backend.TORCHSCRIPT:
        backend_feats = [v for _, v in model_inputs.items()]

    from mmdeploy.codebase.base import BaseBackendModel
    backend_model = BaseBackendModel._build_wrapper(
        backend,
        backend_files,
        device,
        input_names=input_names,
        output_names=output_names)
    with torch.no_grad():
        backend_outputs = backend_model(backend_feats)
    backend_outputs = backend_model.output_to_list(backend_outputs)
    return backend_outputs


def get_rewrite_outputs(wrapped_model: nn.Module,
                        model_inputs: Dict[str, Union[Tuple, List,
                                                      torch.Tensor]],
                        deploy_cfg: Config,
                        run_with_backend: bool = True) -> Tuple[Any, bool]:
    """To get outputs of generated onnx model after rewrite.

    Args:
        wrapped_model (nn.Module): The input model.
        model_inputs (dict): Inputs for model.
        deploy_cfg (Config): Deployment config.
        run_with_backend (bool): Whether to run inference with backend.
            Default is True.

    Returns:
        List[torch.Tensor]: The outputs of model.
        bool: A flag indicate the type of outputs. If the flag is True, then
        the outputs are backend output, otherwise they are outputs of wrapped
        pytorch model.
    """
    backend = get_backend(deploy_cfg)
    with RewriterContext(
            cfg=deploy_cfg, backend=backend.value, opset=11), torch.no_grad():
        ctx_outputs = wrapped_model(**copy.deepcopy(model_inputs))

    ir_type = get_ir_config(deploy_cfg).get('type', None)
    if ir_type == IR.TORCHSCRIPT.value:
        ir_file_path = get_ts_model(wrapped_model, model_inputs, deploy_cfg)
    else:  # TODO onnx as default, make it strict when more IR types involved
        ir_file_path = get_onnx_model(wrapped_model, model_inputs, deploy_cfg)

    backend_outputs = None
    if run_with_backend:
        backend_outputs = get_backend_outputs(ir_file_path, model_inputs,
                                              deploy_cfg)

    if backend_outputs is None:
        return ctx_outputs, False
    else:
        return backend_outputs, True


def get_random_name(
        length: int = 10,
        seed: Optional[Union[int, float, str, bytes,
                             bytearray]] = None) -> str:
    """Generates a random string of the specified length. Can be used to
    generate random names for model inputs and outputs.

    Args:
        length (int): The number of characters in the string..
        seed (Optional[Union[int, float, str, bytes, bytearray]]):
            Seed for a random number generator.

    Returns:
        str: A randomly generated string that can be used as a name.
    """
    if seed:
        random_state = random.getstate()
        random.seed(seed)
    random_name = ''.join(
        random.choices(string.ascii_letters + string.digits, k=length))
    if seed:
        random.setstate(random_state)
    return random_name
