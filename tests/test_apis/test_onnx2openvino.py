# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine import Config

from mmdeploy.utils import Backend
from mmdeploy.utils.test import backend_checker, get_random_name


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


def generate_onnx_file(model, export_img, onnx_file, input_name, output_name):
    with torch.no_grad():
        dynamic_axes = {
            input_name: {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            output_name: {
                0: 'batch'
            }
        }
        torch.onnx.export(
            model,
            export_img,
            onnx_file,
            output_names=[output_name],
            input_names=[input_name],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)
        assert osp.exists(onnx_file)


def get_outputs(pytorch_model, openvino_model_path, input, input_name,
                output_name):
    output_pytorch = pytorch_model(input).numpy()

    from mmdeploy.backend.openvino import OpenVINOWrapper
    openvino_model = OpenVINOWrapper(openvino_model_path)
    openvino_output = openvino_model({input_name: input})[output_name]

    return output_pytorch, openvino_output


def get_base_deploy_cfg():
    deploy_cfg = Config(dict(backend_config=dict(type='openvino')))
    return deploy_cfg


def get_deploy_cfg_with_mo_args():
    deploy_cfg = Config(dict(backend_config=dict(type='openvino')))
    return deploy_cfg


@pytest.mark.parametrize('get_deploy_cfg',
                         [get_base_deploy_cfg, get_deploy_cfg_with_mo_args])
@backend_checker(Backend.OPENVINO)
def test_onnx2openvino(get_deploy_cfg):
    from mmdeploy.apis.openvino import (from_onnx, get_mo_options_from_cfg,
                                        get_output_model_file)
    pytorch_model = TestModel().eval()
    export_img = torch.rand([1, 3, 8, 8])
    onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
    input_name = get_random_name()
    output_name = get_random_name()
    generate_onnx_file(pytorch_model, export_img, onnx_file, input_name,
                       output_name)

    input_info = {input_name: export_img.shape}
    output_names = [output_name]
    openvino_dir = tempfile.TemporaryDirectory().name
    deploy_cfg = get_deploy_cfg()
    mo_options = get_mo_options_from_cfg(deploy_cfg)
    from_onnx(onnx_file, openvino_dir, input_info, output_names, mo_options)
    openvino_model_path = get_output_model_file(onnx_file, openvino_dir)
    assert osp.exists(openvino_model_path), \
        'The file (.xml) for OpenVINO IR has not been created.'

    test_img = torch.rand([1, 3, 16, 16])
    output_pytorch, openvino_output = get_outputs(pytorch_model,
                                                  openvino_model_path,
                                                  test_img, input_name,
                                                  output_name)
    assert np.allclose(output_pytorch, openvino_output), \
        'OpenVINO and PyTorch outputs are not the same.'


@backend_checker(Backend.OPENVINO)
def test_get_input_info_from_cfg():
    from mmdeploy.apis.openvino import get_input_info_from_cfg

    # Test 1
    deploy_cfg = Config()
    with pytest.raises(KeyError):
        get_input_info_from_cfg(deploy_cfg)

    # Test 2
    input_name = 'input'
    height, width = 600, 1000
    shape = [1, 3, height, width]
    expected_input_info = {input_name: shape}
    deploy_cfg = Config({
        'backend_config': {
            'model_inputs': [{
                'opt_shapes': expected_input_info
            }]
        }
    })
    input_info = get_input_info_from_cfg(deploy_cfg)
    assert input_info == expected_input_info, 'Test 2: ' \
        'The expected value of \'input_info\' does not match the received one.'

    # Test 3
    # The case where the input name in 'onnx_config'
    # is different from 'backend_config'.
    onnx_config_input_name = get_random_name(1234)
    deploy_cfg.merge_from_dict(
        {'onnx_config': {
            'input_names': [onnx_config_input_name]
        }})
    expected_input_info = {onnx_config_input_name: shape}
    input_info = get_input_info_from_cfg(deploy_cfg)
    assert input_info == expected_input_info, 'Test 3: ' \
        'The expected value of \'input_info\' does not match the received one.'

    # Test 4
    # The case where 'backend_config.model_inputs.opt_shapes'
    # is given by a list, not a dictionary.
    deploy_cfg.merge_from_dict(
        {'backend_config': {
            'model_inputs': [{
                'opt_shapes': [shape]
            }]
        }})
    input_info = get_input_info_from_cfg(deploy_cfg)
    assert input_info == expected_input_info, 'Test 4: ' \
        'The expected value of \'input_info\' does not match the received one.'
