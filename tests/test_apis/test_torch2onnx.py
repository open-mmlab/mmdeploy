# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import onnx
import pytest
import torch
import torch.nn as nn

from mmdeploy.apis import torch2onnx_impl
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
    return mmcv.Config(
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
            codebase_config=dict(type='mmedit', task=''),
            backend_config=dict(type='onnxruntime')))


@pytest.mark.parametrize('input_name', [input_name])
@pytest.mark.parametrize('output_name', [output_name])
@pytest.mark.parametrize('dynamic_axes',
                         [dynamic_axes_dict, dynamic_axes_list])
def test_torch2onnx(input_name, output_name, dynamic_axes):
    deploy_cfg = get_deploy_cfg(input_name, output_name, dynamic_axes)
    torch2onnx_impl(test_model, test_img, deploy_cfg, onnx_file)

    assert osp.exists(onnx_file)

    model = onnx.load(onnx_file)
    assert model is not None
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        assert False
