# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import pytest
import torch
import torch.nn as nn

from mmdeploy.backend.ncnn.onnx2ncnn import get_output_model_file
from mmdeploy.utils import Backend
from mmdeploy.utils.test import backend_checker

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
test_img = torch.rand([1, 3, 8, 8])


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


test_model = TestModel().eval()


def generate_onnx_file(model):
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
            test_img,
            onnx_file,
            output_names=['output'],
            input_names=['input'],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)
        assert osp.exists(onnx_file)


@backend_checker(Backend.NCNN)
def test_onnx2ncnn():
    from mmdeploy.apis.ncnn import from_onnx
    model = test_model
    generate_onnx_file(model)

    work_dir, _ = osp.split(onnx_file)
    save_param, save_bin = get_output_model_file(onnx_file, work_dir=work_dir)
    file_name = osp.splitext(onnx_file)[0]
    from_onnx(onnx_file, osp.join(work_dir, file_name))
    assert osp.exists(work_dir)
    assert osp.exists(save_param)
    assert osp.exists(save_bin)
