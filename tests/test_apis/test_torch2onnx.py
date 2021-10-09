import os.path as osp
import tempfile

import mmcv
import onnx
import pytest
import torch
import torch.nn as nn

from mmdeploy.apis import torch2onnx_impl

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


test_model = TestModel().eval().cuda()
test_img = torch.rand([1, 3, 8, 8])


def get_deploy_cfg():
    return mmcv.Config(
        dict(
            onnx_config=dict(
                dynamic_axes={
                    'input': {
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    'output': {
                        0: 'batch'
                    }
                },
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                input_names=['input'],
                output_names=['output'],
                input_shape=None),
            codebase_config=dict(type='mmedit', task=''),  # useless
            backend_config=dict(type='onnxruntime')  # useless
        ))


def test_torch2onnx():
    deploy_cfg = get_deploy_cfg()
    torch2onnx_impl(test_model, test_img, deploy_cfg, onnx_file)

    assert osp.exists(onnx_file)

    model = onnx.load(onnx_file)
    assert model is not None
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        assert False
