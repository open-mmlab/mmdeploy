import subprocess
import tempfile

import pytest
import torch
import torch.nn as nn

from mmdeploy.utils.constants import Backend

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
test_img = torch.rand(1, 3, 8, 8)


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + test_img


model = TestModel().eval()


@pytest.fixture(autouse=True, scope='module')
def generate_onnx_file():
    with torch.no_grad():
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
            dynamic_axes=None)


def check_backend_avaiable(backend):
    if backend == Backend.TENSORRT:
        from mmdeploy.apis.tensorrt import is_available as trt_available
        if not trt_available():
            pytest.skip(
                'TensorRT is not installed or custom ops are not compiled.')
        if not torch.cuda.is_available():
            pytest.skip('CUDA is not available.')
    elif backend == Backend.ONNXRUNTIME:
        from mmdeploy.apis.onnxruntime import is_available as ort_available
        if not ort_available():
            pytest.skip(
                'ONNXRuntime is not installed or custom ops are not compiled.')
    elif backend == Backend.PPL:
        from mmdeploy.apis.ppl import is_available as ppl_avaiable
        if not ppl_avaiable():
            pytest.skip('PPL is not available.')
    elif backend == Backend.NCNN:
        from mmdeploy.apis.ncnn import is_available as ncnn_available
        if not ncnn_available():
            pytest.skip(
                'NCNN is not installed or custom ops are not compiled.')
    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import is_available as openvino_available
        if not openvino_available():
            pytest.skip('OpenVINO is not installed.')
    else:
        raise NotImplementedError(f'Unknown backend type: {backend.value}')


def onnx2backend(backend, onnx_file):
    if backend == Backend.TENSORRT:
        from mmdeploy.apis.tensorrt import create_trt_engine, save_trt_engine
        backend_file = tempfile.NamedTemporaryFile(suffix='.engine').name
        engine = create_trt_engine(
            onnx_file, {
                'input': {
                    'min_shape': [1, 3, 8, 8],
                    'opt_shape': [1, 3, 8, 8],
                    'max_shape': [1, 3, 8, 8]
                }
            })
        save_trt_engine(engine, backend_file)
        return backend_file
    elif backend == Backend.ONNXRUNTIME:
        return onnx_file
    elif backend == Backend.PPL:
        from mmdeploy.apis.ppl import onnx2ppl
        algo_file = tempfile.NamedTemporaryFile(suffix='.json').name
        onnx2ppl(algo_file=algo_file, onnx_model=onnx_file)
        return onnx_file, algo_file
    elif backend == Backend.NCNN:
        from mmdeploy.apis.ncnn import get_onnx2ncnn_path
        onnx2ncnn_path = get_onnx2ncnn_path()
        param_file = tempfile.NamedTemporaryFile(suffix='.param').name
        bin_file = tempfile.NamedTemporaryFile(suffix='.bin').name
        subprocess.call([onnx2ncnn_path, onnx_file, param_file, bin_file])
        return param_file, bin_file
    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import onnx2openvino, get_output_model_file
        backend_dir = tempfile.TemporaryDirectory().name
        backend_file = get_output_model_file(onnx_file, backend_dir)
        input_info = {'input': test_img.shape}
        output_names = ['output']
        work_dir = backend_dir
        onnx2openvino(input_info, output_names, onnx_file, work_dir)
        return backend_file


def create_wrapper(backend, model_files):
    if backend == Backend.TENSORRT:
        from mmdeploy.apis.tensorrt import TRTWrapper
        trt_model = TRTWrapper(model_files)
        return trt_model
    elif backend == Backend.ONNXRUNTIME:
        from mmdeploy.apis.onnxruntime import ORTWrapper
        ort_model = ORTWrapper(model_files, 0)
        return ort_model
    elif backend == Backend.PPL:
        from mmdeploy.apis.ppl import PPLWrapper
        ppl_model = PPLWrapper(model_files[0], None, device_id=0)
        return ppl_model
    elif backend == Backend.NCNN:
        from mmdeploy.apis.ncnn import NCNNWrapper
        param_file, bin_file = model_files
        ncnn_model = NCNNWrapper(param_file, bin_file, output_names=['output'])
        return ncnn_model
    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import OpenVINOWrapper
        openvino_model = OpenVINOWrapper(model_files)
        return openvino_model
    else:
        raise NotImplementedError(f'Unknown backend type: {backend.value}')


def run_wrapper(backend, wrapper, input):
    if backend == Backend.TENSORRT:
        input = input.cuda()
        results = wrapper({'input': input})['output']
        results = results.detach().cpu()
        return results
    elif backend == Backend.ONNXRUNTIME:
        input = input.cuda()
        results = wrapper({'input': input})[0]
        return list(results)
    elif backend == Backend.PPL:
        input = input.cuda()
        results = wrapper({'input': input})[0]
        return list(results)
    elif backend == Backend.NCNN:
        input = input.float()
        results = wrapper({'input': input})['output']
        results = results.detach().cpu().numpy()
        results_list = list(results)
        return results_list
    elif backend == Backend.OPENVINO:
        results = wrapper({'input': input})['output']
        return results
    else:
        raise NotImplementedError(f'Unknown backend type: {backend.value}')


ALL_BACKEND = [
    Backend.TENSORRT, Backend.ONNXRUNTIME, Backend.PPL, Backend.NCNN,
    Backend.OPENVINO
]


@pytest.mark.parametrize('backend', ALL_BACKEND)
def test_wrapper(backend):
    check_backend_avaiable(backend)
    model_files = onnx2backend(backend, onnx_file)
    assert model_files is not None
    wrapper = create_wrapper(backend, model_files)
    assert wrapper is not None
    results = run_wrapper(backend, wrapper, test_img)
    assert results is not None
