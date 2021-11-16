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

model_cfg = 'tests/test_mmcls/data/model.py'
deploy_cfg = mmcv.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmcls', task='Classification'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['output'])))
input_img = torch.rand(1, 3, 64, 64)
input = {'img': input_img}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_TensorRTClassifier():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.tensorrt.tensorrt_utils import TRTWrapper
    trt_apis.__dict__.update({'TRTWrapper': TRTWrapper})

    # simplify backend inference
    outputs = {
        'output': torch.rand(1, 3, 64, 64).cuda(),
    }

    with SwitchBackendWrapper(TRTWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmcls.apis.inference import TensorRTClassifier
        trt_classifier = TensorRTClassifier('', [''], 0)
        imgs = torch.rand(1, 3, 64, 64).cuda()

        results = trt_classifier.forward(imgs, return_loss=False)
        assert results is not None, ('failed to get output using '
                                     'TensorRTClassifier')


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_ONNXRuntimeClassifier():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    outputs = torch.rand(1, 3, 64, 64)

    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmcls.apis.inference import ONNXRuntimeClassifier
        ort_classifier = ONNXRuntimeClassifier('', [''], 0)
        imgs = torch.rand(1, 3, 64, 64)

        results = ort_classifier.forward(imgs, return_loss=False)
        assert results is not None, 'failed to get output using '\
            'ONNXRuntimeClassifier'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('pyppl'), reason='requires pyppl')
def test_PPLClassifier():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.ppl.ppl_utils import PPLWrapper
    ppl_apis.__dict__.update({'PPLWrapper': PPLWrapper})

    # simplify backend inference
    outputs = torch.rand(1, 3, 64, 64)

    with SwitchBackendWrapper(PPLWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmcls.apis.inference import PPLClassifier
        ppl_classifier = PPLClassifier('', '', [''], 0)
        imgs = torch.rand(1, 3, 64, 64)

        results = ppl_classifier.forward(imgs, return_loss=False)
        assert results is not None, 'failed to get output using PPLClassifier'


@pytest.mark.skipif(
    not importlib.util.find_spec('ncnn'), reason='requires ncnn')
def test_NCNNClassifier():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.ncnn.ncnn_utils import NCNNWrapper
    ncnn_apis.__dict__.update({'NCNNWrapper': NCNNWrapper})

    # simplify backend inference
    outputs = {'output': torch.rand(1, 3, 64, 64)}

    with SwitchBackendWrapper(NCNNWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmcls.apis.inference import NCNNClassifier
        ncnn_classifier = NCNNClassifier('', '', [''], 0)
        imgs = torch.rand(1, 3, 64, 64)

        results = ncnn_classifier.forward(imgs, return_loss=False)
        assert results is not None, 'failed to get output using NCNNClassifier'


def test_init_pytorch_model():
    model = api_utils.init_pytorch_model(
        Codebase.MMCLS, model_cfg=model_cfg, device='cpu')
    assert model is not None


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def create_backend_model():
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference

    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs=[[1]], model_cfg=model_cfg, deploy_cfg=deploy_cfg)
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
    result = api_utils.run_inference(Codebase.MMCLS, input, model)
    assert result is not None

    # Recovery
    wrapper.recover()


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_visualize():
    numpy_img = np.random.rand(64, 64, 3)
    model, wrapper = create_backend_model()
    result = api_utils.run_inference(Codebase.MMCLS, input, model)
    with tempfile.TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        api_utils.visualize(Codebase.MMCLS, numpy_img, result, model, filename,
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
