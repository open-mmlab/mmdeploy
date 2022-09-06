# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import random
import string
import tempfile
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
import pytest
import torch
from torch import nn

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
    is_custom_ops_available = None
    if backend == Backend.ONNXRUNTIME:
        from mmdeploy.apis.onnxruntime import is_available
        if require_plugin:
            from mmdeploy.apis.onnxruntime import is_custom_ops_available
    elif backend == Backend.TENSORRT:
        from mmdeploy.apis.tensorrt import is_available
        if require_plugin:
            from mmdeploy.apis.tensorrt import is_custom_ops_available
    elif backend == Backend.PPLNN:
        from mmdeploy.apis.pplnn import is_available
    elif backend == Backend.NCNN:
        from mmdeploy.apis.ncnn import is_available
        if require_plugin:
            from mmdeploy.apis.ncnn import is_custom_ops_available
    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import is_available
    elif backend == Backend.RKNN:
        # device not require as backend is not really running
        from mmdeploy.apis.rknn import is_available
    elif backend == Backend.ASCEND:
        from mmdeploy.apis.ascend import is_available
    else:
        warnings.warn('The backend checker is not available')
        return

    checker = pytest.mark.skipif(
        not is_available(), reason=f'{backend.value} package is not available')
    if require_plugin and is_custom_ops_available is not None:
        plugin_checker = pytest.mark.skipif(
            not is_custom_ops_available(),
            reason=f'{backend.value} plugin is not available')

        def double_checker(func):
            func = checker(func)
            func = plugin_checker(func)
            return func

        return double_checker

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
    is_custom_ops_available = None
    if backend == Backend.ONNXRUNTIME:
        from mmdeploy.apis.onnxruntime import is_available
        if require_plugin:
            from mmdeploy.apis.onnxruntime import is_custom_ops_available
    elif backend == Backend.TENSORRT:
        from mmdeploy.apis.tensorrt import is_available
        if require_plugin:
            from mmdeploy.apis.tensorrt import is_custom_ops_available
    elif backend == Backend.PPLNN:
        from mmdeploy.apis.pplnn import is_available
    elif backend == Backend.NCNN:
        from mmdeploy.apis.ncnn import is_available
        if require_plugin:
            from mmdeploy.apis.ncnn import is_custom_ops_available
    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import is_available
    elif backend == Backend.TORCHSCRIPT:
        from mmdeploy.backend.torchscript import ops_available as is_available
    elif backend == Backend.RKNN:
        from mmdeploy.backend.rknn import is_available
        if not is_available():
            # skip CI in github
            pytest.skip(f'{backend.value} package is not available')
        # device required
        from mmdeploy.backend.rknn import device_available as is_available
    elif backend == Backend.ASCEND:
        from mmdeploy.backend.ascend import is_available
    else:
        warnings.warn('The backend checker is not available')
        return

    if not is_available():
        pytest.skip(f'{backend.value} package is not available')
    if require_plugin and is_custom_ops_available is not None:
        if not is_custom_ops_available():
            pytest.skip(f'{backend.value} plugin is not available')


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


class DummyModel(torch.nn.Module):
    """A dummy model for unit tests.

    Args:
        outputs (Any): Predefined output variables.
    """

    def __init__(self, outputs=None, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.outputs = outputs

    def forward(self, *args, **kwargs):
        """Run forward."""
        return self.outputs

    def __call__(self, *args, **kwds):
        """Call the forward method."""
        return self.forward(*args, **kwds)


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
            torch.testing.assert_allclose(
                actual[i], expected[i], rtol=1e-03, atol=1e-05)
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
    model_outputs = func(**model_inputs)
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
                   deploy_cfg: mmcv.Config) -> str:
    """To get path to onnx model after export.

    Args:
        wrapped_model (nn.Module): The input model.
        model_inputs (dict): Inputs for model.
        deploy_cfg (mmcv.Config): Deployment config.

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
                 deploy_cfg: mmcv.Config) -> str:
    """To get path to onnx model after export.

    Args:
        wrapped_model (nn.Module): The input model.
        model_inputs (dict): Inputs for model.
        deploy_cfg (mmcv.Config): Deployment config.

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
                        deploy_cfg: mmcv.Config) -> Union[Any, None]:
    """To get backend outputs of model.

    Args:
        ir_file_path (str): The path to the IR file.
        model_inputs (dict): Inputs for model.
        deploy_cfg (mmcv.Config): Deployment config.

    Returns:
        Union[Any, None]: The outputs of model, decided by the backend wrapper.
            If the backend specified in 'deploy_cfg' is not available,
            then None will be returned.
    """
    backend = get_backend(deploy_cfg)
    flatten_model_inputs = get_flatten_inputs(model_inputs)
    ir_config = get_ir_config(deploy_cfg)
    input_names = ir_config.get('input_names', None)
    output_names = ir_config.get('output_names', None)
    if input_names is None:
        input_names = [
            k for k, v in flatten_model_inputs.items() if k != 'ctx'
        ]

    # prepare backend model and input features
    if backend == Backend.TENSORRT:
        # convert to engine
        import mmdeploy.apis.tensorrt as trt_apis
        if not (trt_apis.is_available()
                and trt_apis.is_custom_ops_available()):
            return None
        trt_file_path = tempfile.NamedTemporaryFile(suffix='.engine').name
        trt_apis.onnx2tensorrt(
            '',
            trt_file_path,
            0,
            deploy_cfg=deploy_cfg,
            onnx_model=ir_file_path)
        backend_files = [trt_file_path]
        for k, v in model_inputs.items():
            model_inputs[k] = model_inputs[k].cuda()

        backend_feats = model_inputs
        device = 'cuda:0'
    elif backend == Backend.ONNXRUNTIME:
        import mmdeploy.apis.onnxruntime as ort_apis
        if not (ort_apis.is_available()
                and ort_apis.is_custom_ops_available()):
            return None
        feature_list = []
        backend_feats = {}
        for k, item in model_inputs.items():
            if type(item) is torch.Tensor:
                feature_list.append(item)
            elif type(item) is tuple or list:
                for i in item:
                    assert type(i) is torch.Tensor, 'model_inputs contains '
                    'nested sequence of torch.Tensor'
                    feature_list.append(i)
            else:
                raise TypeError('values of model_inputs are expected to be '
                                'torch.Tensor or its sequence, '
                                f'but got {type(model_inputs)}')

        # for onnx file generated with list[torch.Tensor] input,
        # the input dict keys are just numbers if not specified
        for i in range(len(feature_list)):
            if i < len(input_names):
                backend_feats[input_names[i]] = feature_list[i]
            else:
                backend_feats[str(i)] = feature_list[i]
        backend_files = [ir_file_path]
        device = 'cpu'
    elif backend == Backend.NCNN:
        import mmdeploy.apis.ncnn as ncnn_apis
        if not (ncnn_apis.is_available()
                and ncnn_apis.is_custom_ops_available()):
            return None
        param_path, bin_path = ncnn_apis.get_output_model_file(ir_file_path)
        ncnn_files_prefix = osp.splitext(ir_file_path)[0]
        ncnn_apis.from_onnx(ir_file_path, ncnn_files_prefix)
        backend_files = [param_path, bin_path]
        backend_feats = flatten_model_inputs
        device = 'cpu'

    elif backend == Backend.OPENVINO:
        import mmdeploy.apis.openvino as openvino_apis
        if not openvino_apis.is_available():
            return None
        from mmdeploy.apis.openvino import get_mo_options_from_cfg
        openvino_work_dir = tempfile.TemporaryDirectory().name
        openvino_file_path = openvino_apis.get_output_model_file(
            ir_file_path, openvino_work_dir)
        input_info = {
            name: value.shape
            for name, value in flatten_model_inputs.items()
        }
        mo_options = get_mo_options_from_cfg(deploy_cfg)
        openvino_apis.from_onnx(ir_file_path, openvino_work_dir, input_info,
                                output_names, mo_options)
        backend_files = [openvino_file_path]
        backend_feats = flatten_model_inputs
        device = 'cpu'

    elif backend == Backend.DEFAULT:
        return None
    elif backend == Backend.TORCHSCRIPT:
        backend_files = [ir_file_path]
        device = 'cpu'
        backend_feats = [v for _, v in model_inputs.items()]
    elif backend == Backend.ASCEND:
        # Ascend model conversion
        import mmdeploy.apis.ascend as ascend_apis
        from mmdeploy.utils import get_model_inputs
        if not ascend_apis.is_available():
            return None
        work_dir = osp.split(ir_file_path)[0]
        # convert model
        convert_args = get_model_inputs(deploy_cfg)
        ascend_apis.from_onnx(ir_file_path, work_dir, convert_args[0])
        om_file_name = osp.splitext(osp.split(ir_file_path)[1])[0]
        backend_files = [osp.join(work_dir, om_file_name + '.om')]
        backend_feats = flatten_model_inputs
        device = 'cpu'
    else:
        raise NotImplementedError(
            f'Unimplemented backend type: {backend.value}')

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
                        deploy_cfg: mmcv.Config,
                        run_with_backend: bool = True) -> Tuple[Any, bool]:
    """To get outputs of generated onnx model after rewrite.

    Args:
        wrapped_model (nn.Module): The input model.
        model_inputs (dict): Inputs for model.
        deploy_cfg (mmcv.Config): Deployment config.
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
        ctx_outputs = wrapped_model(**model_inputs)

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
