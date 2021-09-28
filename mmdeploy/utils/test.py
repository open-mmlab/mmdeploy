import tempfile
from typing import List, Union

import mmcv
import numpy as np
import torch
from torch import nn

from mmdeploy.core import (RewriterContext, patch_model,
                           register_extra_symbolics)
from mmdeploy.utils import Backend, get_backend, get_onnx_config


class WrapFunction(nn.Module):

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


def assert_allclose(actual: List[Union[torch.Tensor, np.ndarray]],
                    expected: List[Union[torch.Tensor, np.ndarray]],
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


def get_rewrite_outputs(wrapped_model: nn.Module, model_inputs: dict,
                        deploy_cfg: mmcv.Config):
    """To get outputs of generated onnx model after rewrite.

    Args:
        wrap_model (nn.Module): The input model.
        func_name (str): The function of model.
        model_inputs (dict): Inputs for model.

    Returns:
        Any: The outputs of model, decided by the backend wrapper.
    """
    onnx_file_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
    pytorch2onnx_cfg = get_onnx_config(deploy_cfg)
    backend = get_backend(deploy_cfg)
    register_extra_symbolics({}, backend=backend.value, opset=11)
    patched_model = patch_model(
        wrapped_model, cfg=deploy_cfg, backend=backend.value)
    input_names = [k for k, v in model_inputs.items() if k != 'ctx']
    with RewriterContext(
            cfg=deploy_cfg, backend=backend.value), torch.no_grad():
        ctx_outputs = wrapped_model(**model_inputs)
        torch.onnx.export(
            patched_model,
            tuple([v for k, v in model_inputs.items()]),
            onnx_file_path,
            export_params=True,
            input_names=input_names,
            output_names=None,
            opset_version=11,
            dynamic_axes=pytorch2onnx_cfg.get('dynamic_axes', None),
            keep_initializers_as_inputs=False)

    # prepare backend model and input features
    if backend == Backend.TENSORRT:
        # convert to engine
        import mmdeploy.apis.tensorrt as trt_apis
        if not trt_apis.is_available():
            return ctx_outputs
        trt_file_path = tempfile.NamedTemporaryFile(suffix='.engine').name
        trt_apis.onnx2tensorrt(
            '',
            trt_file_path,
            0,
            deploy_cfg=deploy_cfg,
            onnx_model=onnx_file_path)
        backend_model = trt_apis.TRTWrapper(trt_file_path)
        backend_feats = model_inputs
    elif backend == Backend.ONNXRUNTIME:
        import mmdeploy.apis.onnxruntime as ort_apis
        if not ort_apis.is_available():
            return ctx_outputs
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
    with torch.no_grad():
        backend_outputs = backend_model.forward(backend_feats)
    return backend_outputs
