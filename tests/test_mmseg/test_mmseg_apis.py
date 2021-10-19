import importlib
import os
import tempfile

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.apis.ncnn as ncnn_apis
import mmdeploy.apis.onnxruntime as ort_apis
import mmdeploy.apis.ppl as ppl_apis
import mmdeploy.apis.tensorrt as trt_apis
import mmdeploy.apis.utils as api_utils
from mmdeploy.utils.constants import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_TensorRTSegmentor():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.tensorrt.tensorrt_utils import TRTWrapper
    trt_apis.__dict__.update({'TRTWrapper': TRTWrapper})

    # simplify backend inference
    outputs = {
        'output': torch.rand(1, 1, 64, 64).cuda(),
    }
    with SwitchBackendWrapper(TRTWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmseg.apis.inference import TensorRTSegmentor
        trt_segmentor = TensorRTSegmentor('', ['' for i in range(19)],
                                          np.empty([19], dtype=int), 0)
        trt_segmentor.output_name = 'output'
        imgs = torch.rand(1, 3, 64, 64).cuda()
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = trt_segmentor.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using'
        'TensorRTSegmentor'


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_ONNXRuntimeSegmentor():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    outputs = torch.rand(1, 1, 64, 64)
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmseg.apis.inference import ONNXRuntimeSegmentor
        ort_segmentor = ONNXRuntimeSegmentor('', ['' for i in range(19)],
                                             np.empty([19], dtype=int), 0)
        imgs = torch.rand(1, 1, 64, 64)
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = ort_segmentor.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using '
        'ONNXRuntimeSegmentor'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('pyppl'), reason='requires pyppl')
def test_PPLSegmentor():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.ppl.ppl_utils import PPLWrapper
    ppl_apis.__dict__.update({'PPLWrapper': PPLWrapper})

    # simplify backend inference
    outputs = torch.rand(1, 1, 64, 64)

    with SwitchBackendWrapper(PPLWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmseg.apis.inference import PPLSegmentor
        ppl_segmentor = PPLSegmentor('', ['' for i in range(19)],
                                     np.empty([19], dtype=int), 0)
        imgs = torch.rand(1, 3, 64, 64)
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = ppl_segmentor.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using PPLSegmentor'


@pytest.mark.skipif(
    not importlib.util.find_spec('ncnn'), reason='requires ncnn')
def test_NCNNSegmentor():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.ncnn.ncnn_utils import NCNNWrapper
    ncnn_apis.__dict__.update({'NCNNWrapper': NCNNWrapper})

    # simplify backend inference
    outputs = {
        'output': torch.rand(1, 1, 64, 64),
    }

    with SwitchBackendWrapper(NCNNWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmseg.apis.inference import NCNNSegmentor

        ncnn_segmentor = NCNNSegmentor(['', ''], ['' for i in range(19)],
                                       np.empty([19], dtype=int), 0)
        imgs = [torch.rand(1, 3, 32, 32)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'scale_factor': [2.09, 1.87, 2.09, 1.87],
        }]]

        results = ncnn_segmentor.forward(imgs, img_metas)
        assert results is not None, 'failed to get output using NCNNSegmentor'


model_cfg = 'tests/test_mmseg/data/model.py'
deploy_cfg = mmcv.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmseg', task='Segmentation'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['output'])))
input_img = torch.rand(1, 3, 64, 64)
img_metas = [[{
    'ori_shape': [64, 64, 3],
    'img_shape': [64, 64, 3],
    'scale_factor': [2.09, 1.87, 2.09, 1.87],
}]]
input = {'img': input_img, 'img_metas': img_metas}


def test_init_pytorch_model():
    model = api_utils.init_pytorch_model(
        Codebase.MMSEG, model_cfg=model_cfg, device='cpu')
    assert model is not None


def create_backend_model():
    if not importlib.util.find_spec('onnxruntime'):
        pytest.skip('requires onnxruntime')
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
    model = api_utils.init_backend_model([''], model_cfg, deploy_cfg)

    return model, wrapper


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_init_backend_model():
    model, wrapper = create_backend_model()
    assert model is not None

    # Recovery
    wrapper.recover()


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_run_inference():
    model, wrapper = create_backend_model()
    result = api_utils.run_inference(Codebase.MMSEG, input, model)
    assert result is not None
    assert result[0] is not None

    # Recovery
    wrapper.recover()


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_visualize():
    model, wrapper = create_backend_model()
    result = api_utils.run_inference(Codebase.MMSEG, input, model)
    with tempfile.TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        numpy_img = np.random.rand(64, 64, 3)
        api_utils.visualize(Codebase.MMSEG, numpy_img, result, model, filename,
                            Backend.ONNXRUNTIME)
        assert os.path.exists(filename)

    # Recovery
    wrapper.recover()


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_inference_model():
    numpy_img = np.random.rand(64, 64, 3)
    with tempfile.TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        model, wrapper = create_backend_model()
        from mmdeploy.apis.inference import inference_model
        inference_model(model_cfg, deploy_cfg, model, numpy_img, 'cpu',
                        Backend.ONNXRUNTIME, filename, False)
        assert os.path.exists(filename)

    # Recovery
    wrapper.recover()
