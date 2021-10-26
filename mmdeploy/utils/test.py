import tempfile
from typing import Any, Dict, List, Tuple, Union

import mmcv
import numpy as np
import torch
from torch import nn

from mmdeploy.core import RewriterContext, patch_model
from mmdeploy.utils import Backend, get_backend, get_onnx_config


class WrapFunction(nn.Module):
    """Simple wrapper for a function."""

    def __init__(self, wrapped_function, **kwargs):
        super(WrapFunction, self).__init__()
        self.wrapped_function = wrapped_function
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
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
        kwargs.update(self.kwargs)
        func = getattr(self.model, self.func_name)
        return func(*args, **kwargs)


class SwitchBackendWrapper:
    """A switcher for backend wrapper for unit tests.
    Examples:
        >>> from mmdeploy.utils.test import SwitchBackendWrapper
        >>> from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
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
        """A dummy wrapper for unit tests."""

        def __init__(self, *args, **kwargs):
            self.output_names = ['dets', 'labels']

        def forward(self, *args, **kwargs):
            return self.outputs

        def __call__(self, *args, **kwds):
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


def get_model_outputs(model: nn.Module, func_name: str, model_inputs: dict):
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
            for i, tensor in enumerate(value):
                name_i = f'{name}_{i}'
                flatten_inputs[name_i] = tensor
    return flatten_inputs


def get_rewrite_outputs(wrapped_model: nn.Module,
                        model_inputs: Dict[str, Union[Tuple, List,
                                                      torch.Tensor]],
                        deploy_cfg: mmcv.Config) -> Tuple[Any, bool]:
    """To get outputs of generated onnx model after rewrite.

    Args:
        wrap_model (nn.Module): The input model.
        func_name (str): The function of model.
        model_inputs (dict): Inputs for model.

    Returns:
        Any: The outputs of model, decided by the backend wrapper.
        bool: A flag indicate the type of outputs. If the flag is True, then
        the outputs are backend output, otherwise they are outputs of wrapped
        pytorch model.
    """
    onnx_file_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
    pytorch2onnx_cfg = get_onnx_config(deploy_cfg)
    backend = get_backend(deploy_cfg)
    patched_model = patch_model(
        wrapped_model, cfg=deploy_cfg, backend=backend.value)
    flatten_model_inputs = get_flatten_inputs(model_inputs)
    input_names = [k for k, v in flatten_model_inputs.items() if k != 'ctx']
    output_names = pytorch2onnx_cfg.get('output_names', None)
    with RewriterContext(
            cfg=deploy_cfg, backend=backend.value, opset=11), torch.no_grad():
        ctx_outputs = wrapped_model(**model_inputs)
        torch.onnx.export(
            patched_model,
            tuple([v for k, v in model_inputs.items()]),
            onnx_file_path,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=11,
            dynamic_axes=pytorch2onnx_cfg.get('dynamic_axes', None),
            keep_initializers_as_inputs=False)
    # prepare backend model and input features
    if backend == Backend.TENSORRT:
        # convert to engine
        import mmdeploy.apis.tensorrt as trt_apis
        if not trt_apis.is_available():
            return ctx_outputs, False
        trt_file_path = tempfile.NamedTemporaryFile(suffix='.engine').name
        trt_apis.onnx2tensorrt(
            '',
            trt_file_path,
            0,
            deploy_cfg=deploy_cfg,
            onnx_model=onnx_file_path)
        backend_model = trt_apis.TRTWrapper(trt_file_path)
        for k, v in model_inputs.items():
            model_inputs[k] = model_inputs[k].cuda()

        backend_feats = model_inputs
    elif backend == Backend.ONNXRUNTIME:
        import mmdeploy.apis.onnxruntime as ort_apis
        if not ort_apis.is_available():
            return ctx_outputs, False
        backend_model = ort_apis.ORTWrapper(onnx_file_path, 0, None)
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
    elif backend == Backend.NCNN:
        return ctx_outputs, False
    elif backend == Backend.OPENVINO:
        import mmdeploy.apis.openvino as openvino_apis
        if not openvino_apis.is_available():
            return ctx_outputs, False
        openvino_work_dir = tempfile.TemporaryDirectory().name
        openvino_file_path = openvino_apis.get_output_model_file(
            onnx_file_path, openvino_work_dir)
        input_info = {
            name: value.shape
            for name, value in flatten_model_inputs.items()
        }
        openvino_apis.onnx2openvino(input_info, output_names, onnx_file_path,
                                    openvino_work_dir)
        backend_model = openvino_apis.OpenVINOWrapper(openvino_file_path)

        backend_feats = flatten_model_inputs
    elif backend == Backend.DEFAULT:
        return ctx_outputs, False
    else:
        raise NotImplementedError(
            f'Unimplemented backend type: {backend.value}')

    with torch.no_grad():
        backend_outputs = backend_model.forward(backend_feats)
    return backend_outputs, True
