# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from mmdeploy.utils import Backend
from mmdeploy.utils.test import backend_checker


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


def generate_onnx_file(model, export_img, onnx_file):
    with torch.no_grad():
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'output': {
                0: 'batch'
            }
        }
        torch.onnx.export(
            model,
            export_img,
            onnx_file,
            output_names=['output'],
            input_names=['input'],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)
        assert osp.exists(onnx_file)


def get_outputs(pytorch_model, openvino_model_path, input):
    output_pytorch = pytorch_model(input).numpy()

    from mmdeploy.backend.openvino import OpenVINOWrapper
    openvino_model = OpenVINOWrapper(openvino_model_path)
    openvino_output = openvino_model({'input': input})['output']

    return output_pytorch, openvino_output


@backend_checker(Backend.OPENVINO)
def test_onnx2openvino():
    from mmdeploy.apis.openvino import get_output_model_file, onnx2openvino
    pytorch_model = TestModel().eval()
    export_img = torch.rand([1, 3, 8, 8])
    onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
    generate_onnx_file(pytorch_model, export_img, onnx_file)

    input_info = {'input': export_img.shape}
    output_names = ['output']
    openvino_dir = tempfile.TemporaryDirectory().name
    onnx2openvino(input_info, output_names, onnx_file, openvino_dir)
    openvino_model_path = get_output_model_file(onnx_file, openvino_dir)
    assert osp.exists(openvino_model_path), \
        'The file (.xml) for OpenVINO IR has not been created.'

    test_img = torch.rand([1, 3, 16, 16])
    output_pytorch, openvino_output = get_outputs(pytorch_model,
                                                  openvino_model_path,
                                                  test_img)
    assert np.allclose(output_pytorch, openvino_output), \
        'OpenVINO and PyTorch outputs are not the same.'


@backend_checker(Backend.OPENVINO)
def test_can_not_run_onnx2openvino_without_mo():
    current_environ = dict(os.environ)
    os.environ.clear()

    is_error = False
    try:
        from mmdeploy.apis.openvino import onnx2openvino
        onnx2openvino({}, ['output'], 'tmp.onnx', '/tmp')
    except RuntimeError:
        is_error = True

    os.environ.update(current_environ)
    assert is_error, \
        'The onnx2openvino script was launched without checking for MO.'


@backend_checker(Backend.OPENVINO)
def test_get_input_shape_from_cfg():
    from mmdeploy.apis.openvino import get_input_shape_from_cfg

    # Test with default value
    model_cfg = {}
    input_shape = get_input_shape_from_cfg(model_cfg)
    assert input_shape == [1, 3, 800, 1344], \
        'The function returned a different default shape.'

    # Test with config that contains the required data.
    height, width = 800, 1200
    model_cfg = {'test_pipeline': [{}, {'img_scale': (width, height)}]}
    input_shape = get_input_shape_from_cfg(model_cfg)
    assert input_shape == [1, 3, height, width], \
        'The shape in the config does not match the output shape.'
