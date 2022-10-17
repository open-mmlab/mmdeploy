# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import subprocess
import tempfile

import pytest
import torch
import torch.nn as nn

from mmdeploy.utils.constants import Backend
from mmdeploy.utils.test import check_backend

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
ts_file = tempfile.NamedTemporaryFile(suffix='.pt').name
test_img = torch.rand(1, 3, 8, 8)
output_names = ['output']
input_names = ['input']
target_platform = 'rk3588'  # rknn pre-compiled model need device


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
            output_names=output_names,
            input_names=input_names,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=None)


@pytest.fixture(autouse=True, scope='module')
def generate_torchscript_file():
    import mmcv

    backend = Backend.TORCHSCRIPT.value
    deploy_cfg = mmcv.Config({'backend_config': dict(type=backend)})

    from mmdeploy.apis.torch_jit import trace
    context_info = dict(deploy_cfg=deploy_cfg)
    output_prefix = osp.splitext(ts_file)[0]

    example_inputs = torch.rand(1, 3, 8, 8)
    trace(
        model,
        example_inputs,
        output_path_prefix=output_prefix,
        backend=backend,
        context_info=context_info)


def onnx2backend(backend, onnx_file):
    if backend == Backend.TENSORRT:
        from mmdeploy.backend.tensorrt import from_onnx
        backend_file = tempfile.NamedTemporaryFile(suffix='.engine').name
        from_onnx(
            onnx_file,
            osp.splitext(backend_file)[0], {
                'input': {
                    'min_shape': [1, 3, 8, 8],
                    'opt_shape': [1, 3, 8, 8],
                    'max_shape': [1, 3, 8, 8]
                }
            })
        return backend_file
    elif backend == Backend.ONNXRUNTIME:
        return onnx_file
    elif backend == Backend.PPLNN:
        from mmdeploy.apis.pplnn import onnx2pplnn
        algo_file = tempfile.NamedTemporaryFile(suffix='.json').name
        onnx2pplnn(algo_file=algo_file, onnx_model=onnx_file)
        return onnx_file, algo_file
    elif backend == Backend.NCNN:
        from mmdeploy.backend.ncnn.init_plugins import get_onnx2ncnn_path
        onnx2ncnn_path = get_onnx2ncnn_path()
        param_file = tempfile.NamedTemporaryFile(suffix='.param').name
        bin_file = tempfile.NamedTemporaryFile(suffix='.bin').name
        subprocess.call([onnx2ncnn_path, onnx_file, param_file, bin_file])
        return param_file, bin_file
    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import from_onnx, get_output_model_file
        backend_dir = tempfile.TemporaryDirectory().name
        backend_file = get_output_model_file(onnx_file, backend_dir)
        input_info = {'input': test_img.shape}
        output_names = ['output']
        work_dir = backend_dir
        from_onnx(onnx_file, work_dir, input_info, output_names)
        return backend_file
    elif backend == Backend.RKNN:
        import mmcv

        from mmdeploy.apis.rknn import onnx2rknn
        rknn_file = onnx_file.replace('.onnx', '.rknn')
        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(
                    type='rknn',
                    common_config=dict(target_platform=target_platform),
                    quantization_config=dict(
                        do_quantization=False, dataset=None),
                    input_size_list=[[3, 8, 8]])))
        onnx2rknn(onnx_file, rknn_file, deploy_cfg)
        return rknn_file
    elif backend == Backend.ASCEND:
        import mmcv

        from mmdeploy.apis.ascend import from_onnx
        backend_dir = tempfile.TemporaryDirectory().name
        work_dir = backend_dir
        file_name = osp.splitext(osp.split(onnx_file)[1])[0]
        backend_file = osp.join(work_dir, file_name + '.om')
        model_inputs = mmcv.Config(
            dict(input_shapes=dict(input=test_img.shape)))
        from_onnx(onnx_file, work_dir, model_inputs)
        return backend_file


def create_wrapper(backend, model_files):
    if backend == Backend.TENSORRT:
        from mmdeploy.backend.tensorrt import TRTWrapper
        trt_model = TRTWrapper(model_files, output_names)
        return trt_model
    elif backend == Backend.ONNXRUNTIME:
        from mmdeploy.backend.onnxruntime import ORTWrapper
        ort_model = ORTWrapper(model_files, 'cpu', output_names)
        return ort_model
    elif backend == Backend.PPLNN:
        from mmdeploy.backend.pplnn import PPLNNWrapper
        onnx_file, algo_file = model_files
        pplnn_model = PPLNNWrapper(onnx_file, algo_file, 'cpu', output_names)
        return pplnn_model
    elif backend == Backend.NCNN:
        from mmdeploy.backend.ncnn import NCNNWrapper
        param_file, bin_file = model_files
        ncnn_model = NCNNWrapper(param_file, bin_file, output_names)
        return ncnn_model
    elif backend == Backend.OPENVINO:
        from mmdeploy.backend.openvino import OpenVINOWrapper
        openvino_model = OpenVINOWrapper(model_files, output_names)
        return openvino_model
    elif backend == Backend.TORCHSCRIPT:
        from mmdeploy.backend.torchscript import TorchscriptWrapper
        torchscript_model = TorchscriptWrapper(
            model_files, input_names=input_names, output_names=output_names)
        return torchscript_model
    elif backend == Backend.RKNN:
        from mmdeploy.backend.rknn import RKNNWrapper
        rknn_model = RKNNWrapper(
            model_files,
            common_config=dict(target_platform=target_platform),
            output_names=output_names)
        return rknn_model
    elif backend == Backend.ASCEND:
        from mmdeploy.backend.ascend import AscendWrapper
        ascend_model = AscendWrapper(model_files)
        return ascend_model
    else:
        raise NotImplementedError(f'Unknown backend type: {backend.value}')


def run_wrapper(backend, wrapper, input):
    if backend == Backend.TENSORRT:
        input = input.cuda()
        results = wrapper({'input': input})['output']
        results = results.detach().cpu()
        return results
    elif backend == Backend.ONNXRUNTIME:
        results = wrapper({'input': input})['output']
        results = results.detach().cpu()
        return results
    elif backend == Backend.PPLNN:
        results = wrapper({'input': input})['output']
        results = results.detach().cpu()
        return results
    elif backend == Backend.NCNN:
        input = input.float()
        results = wrapper({'input': input})['output']
        results = results.detach().cpu()
        return results
    elif backend == Backend.OPENVINO:
        results = wrapper({'input': input})['output']
        results = results.detach().cpu()
        return results
    elif backend == Backend.TORCHSCRIPT:
        results = wrapper({'input': input})['output']
        return results
    elif backend == Backend.RKNN:
        results = wrapper({'input': input})
    elif backend == Backend.ASCEND:
        results = wrapper({'input': input})['output']
        return results
    else:
        raise NotImplementedError(f'Unknown backend type: {backend.value}')


ALL_BACKEND = [
    Backend.TENSORRT, Backend.ONNXRUNTIME, Backend.PPLNN, Backend.NCNN,
    Backend.OPENVINO, Backend.TORCHSCRIPT, Backend.ASCEND, Backend.RKNN
]


@pytest.mark.parametrize('backend', ALL_BACKEND)
def test_wrapper(backend):
    check_backend(backend, True)
    if backend == Backend.TORCHSCRIPT:
        model_files = ts_file
    else:
        model_files = onnx2backend(backend, onnx_file)
    assert model_files is not None
    wrapper = create_wrapper(backend, model_files)
    assert wrapper is not None
    results = run_wrapper(backend, wrapper, test_img)
    assert results is not None
