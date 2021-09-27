import tempfile

import pytest
import torch
import torch.nn as nn

from mmdeploy.apis.tensorrt import (TRTWrapper, create_trt_engine,
                                    save_trt_engine)
from mmdeploy.utils.constants import Backend

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
test_img = torch.rand([1, 3, 64, 64])


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


model = TestModel().eval().cuda()


@pytest.fixture(autouse=True, scope='module')
def generate_onnx_file():
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


def check_backend_avaiable(backend):
    if backend == Backend.TENSORRT:
        from mmdeploy.apis.tensorrt import is_available as trt_available
        if not trt_available():
            pytest.skip(
                'TensorRT is not installed or custom ops are not compiled.')
        if not torch.cuda.is_available():
            pytest.skip('CUDA is not available.')


def onnx2backend(backend, onnx_file):

    if backend == Backend.TENSORRT:
        backend_file = tempfile.NamedTemporaryFile(suffix='.engine').name
        engine = create_trt_engine(
            onnx_file, {
                'input': {
                    'min_shape': [1, 3, 64, 64],
                    'opt_shape': [1, 3, 64, 64],
                    'max_shape': [1, 3, 64, 64]
                }
            })
        save_trt_engine(engine, backend_file)
        return backend_file


def create_wrapper(backend, engine_file):
    if backend == Backend.TENSORRT:
        trt_model = TRTWrapper(engine_file)
        return trt_model


def run_wrapper(backend, wrapper, input):
    if backend == Backend.TENSORRT:
        input = input.cuda()
        results = wrapper({'input': input})['output']
        results = results.detach().cpu()
        return results


ALL_BACKEND = [Backend.TENSORRT]


@pytest.mark.parametrize('backend', ALL_BACKEND)
def test_wrapper(backend):
    check_backend_avaiable(backend)
    model_files = onnx2backend(backend, onnx_file)
    assert model_files is not None
    wrapper = create_wrapper(backend, model_files)
    assert wrapper is not None
    results = run_wrapper(backend, wrapper, test_img)
    assert results is not None
