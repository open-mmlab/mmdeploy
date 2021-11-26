import os.path as osp
import tempfile

import pytest
import torch
import torch.nn as nn

from mmdeploy.apis.ncnn import is_available
from mmdeploy.backend.ncnn.onnx2ncnn import get_output_model_file

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
test_img = torch.rand([1, 3, 8, 8])

ncnn_skip = not is_available()


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


@pytest.mark.skipif(ncnn_skip, reason='ncnn not avaiable')
def test_onnx2ncnn():
    from mmdeploy.apis.ncnn import onnx2ncnn
    model = test_model
    generate_onnx_file(model)

    work_dir, _ = osp.split(onnx_file)
    save_param, save_bin = get_output_model_file(onnx_file, work_dir=work_dir)
    onnx2ncnn(onnx_file, save_param, save_bin)
    assert osp.exists(work_dir)
    assert osp.exists(save_param)
    assert osp.exists(save_bin)
