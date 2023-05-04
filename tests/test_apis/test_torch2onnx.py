# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import onnx
import pytest
import torch
import torch.nn as nn
from mmengine import Config

from mmdeploy.apis.onnx import export
from mmdeploy.utils.config_utils import (get_backend, get_dynamic_axes,
                                         get_onnx_config)
from mmdeploy.utils.test import get_random_name

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


test_model = TestModel().eval().cuda()
test_img = torch.rand([1, 3, 8, 8])
input_name = get_random_name()
output_name = get_random_name()
dynamic_axes_dict = {
    input_name: {
        0: 'batch',
        2: 'height',
        3: 'width'
    },
    output_name: {
        0: 'batch'
    }
}
dynamic_axes_list = [[0, 2, 3], [0]]


def get_deploy_cfg(input_name, output_name, dynamic_axes):
    return Config(
        dict(
            onnx_config=dict(
                dynamic_axes=dynamic_axes,
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                input_names=[input_name],
                output_names=[output_name],
                input_shape=None),
            codebase_config=dict(type='mmagic', task=''),
            backend_config=dict(type='onnxruntime')))


@pytest.mark.parametrize('input_name', [input_name])
@pytest.mark.parametrize('output_name', [output_name])
@pytest.mark.parametrize('dynamic_axes',
                         [dynamic_axes_dict, dynamic_axes_list])
def test_torch2onnx(input_name, output_name, dynamic_axes):
    deploy_cfg = get_deploy_cfg(input_name, output_name, dynamic_axes)

    output_prefix = osp.splitext(onnx_file)[0]
    context_info = dict(cfg=deploy_cfg)
    backend = get_backend(deploy_cfg).value
    onnx_cfg = get_onnx_config(deploy_cfg)
    opset_version = onnx_cfg.get('opset_version', 11)

    input_names = onnx_cfg['input_names']
    output_names = onnx_cfg['output_names']
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
    verbose = not onnx_cfg.get('strip_doc_string', True) or onnx_cfg.get(
        'verbose', False)
    keep_initializers_as_inputs = onnx_cfg.get('keep_initializers_as_inputs',
                                               True)
    export(
        test_model,
        test_img,
        context_info=context_info,
        output_path_prefix=output_prefix,
        backend=backend,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
        keep_initializers_as_inputs=keep_initializers_as_inputs)

    assert osp.exists(onnx_file)

    model = onnx.load(onnx_file)
    assert model is not None
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        assert False
