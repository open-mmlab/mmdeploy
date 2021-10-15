import importlib
import os
import tempfile

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.apis.onnxruntime as ort_apis
import mmdeploy.apis.ppl as ppl_apis
import mmdeploy.apis.tensorrt as trt_apis
import mmdeploy.apis.test as api_test
import mmdeploy.apis.utils as api_utils
from mmdeploy.utils.constants import Backend, Codebase
from mmdeploy.utils.test import SwitchBackendWrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_TensorRTRestorer():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.tensorrt.tensorrt_utils import TRTWrapper
    trt_apis.__dict__.update({'TRTWrapper': TRTWrapper})

    # simplify backend inference
    outputs = {
        'output': torch.rand(1, 3, 64, 64).cuda(),
    }

    with SwitchBackendWrapper(TRTWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmedit.apis.inference import TensorRTRestorer
        trt_restorer = TensorRTRestorer('', 0)
        imgs = torch.rand(1, 3, 64, 64).cuda()

        results = trt_restorer.forward(imgs)
        assert results is not None, ('failed to get output using '
                                     'TensorRTRestorer')

        results = trt_restorer.forward(imgs, test_mode=True)
        assert results is not None, ('failed to get output using '
                                     'TensorRTRestorer')


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_ONNXRuntimeRestorer():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    outputs = torch.rand(1, 3, 64, 64)

    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmedit.apis.inference import ONNXRuntimeRestorer
        ort_restorer = ONNXRuntimeRestorer('', 0)
        imgs = torch.rand(1, 3, 64, 64)

        results = ort_restorer.forward(imgs)
        assert results is not None, 'failed to get output using '\
            'ONNXRuntimeRestorer'

        results = ort_restorer.forward(imgs, test_mode=True)
        assert results is not None, 'failed to get output using '\
            'ONNXRuntimeRestorer'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('pyppl'), reason='requires pyppl')
def test_PPLRestorer():
    # force add backend wrapper regardless of plugins
    from mmdeploy.apis.ppl.ppl_utils import PPLWrapper
    ppl_apis.__dict__.update({'PPLWrapper': PPLWrapper})

    # simplify backend inference
    outputs = torch.rand(1, 3, 64, 64)

    with SwitchBackendWrapper(PPLWrapper) as wrapper:
        wrapper.set(outputs=outputs)

        from mmdeploy.mmedit.apis.inference import PPLRestorer
        ppl_restorer = PPLRestorer('', 0)
        imgs = torch.rand(1, 3, 64, 64)

        results = ppl_restorer.forward(imgs)
        assert results is not None, 'failed to get output using PPLRestorer'

        results = ppl_restorer.forward(imgs, test_mode=True)
        assert results is not None, 'failed to get output using PPLRestorer'


model_cfg = 'tests/test_mmedit/data/model.py'
deploy_cfg = mmcv.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmedit', task='SuperResolution'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['output'])))
input_img = torch.rand(1, 3, 64, 64)
input = {'lq': input_img}


def test_init_pytorch_model():
    model = api_utils.init_pytorch_model(
        Codebase.MMEDIT, model_cfg=model_cfg, device='cpu')
    assert model is not None


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def create_backend_model():
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
    result = api_utils.run_inference(Codebase.MMEDIT, input, model)
    assert isinstance(result, np.ndarray)

    # Recovery
    wrapper.recover()


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_visualize():
    model, wrapper = create_backend_model()
    result = api_utils.run_inference(Codebase.MMEDIT, input, model)
    with tempfile.TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        api_utils.visualize(Codebase.MMEDIT, input, result, model, filename,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
def test_test():
    from mmcv.parallel import MMDataParallel
    with tempfile.TemporaryDirectory() as dir:

        # Export a complete model
        numpy_img = np.random.rand(50, 50, 3)
        onnx_filename = 'end2end.onnx'
        onnx_path = os.path.join(dir, onnx_filename)
        from mmdeploy.apis import torch2onnx
        torch2onnx(numpy_img, dir, onnx_filename, deploy_cfg, model_cfg)
        assert os.path.exists(onnx_path)

        # Prepare dataloader
        dataset = api_utils.build_dataset(
            Codebase.MMEDIT, model_cfg, dataset_type='test')
        assert dataset is not None, 'Failed to build dataset'
        dataloader = api_utils.build_dataloader(Codebase.MMEDIT, dataset, 1, 1)
        assert dataloader is not None, 'Failed to build dataloader'

        # Prepare model
        model = api_utils.init_backend_model([onnx_path], model_cfg,
                                             deploy_cfg)
        model = MMDataParallel(model, device_ids=[0])
        assert model is not None

        # Run test
        outputs = api_test.single_gpu_test(Codebase.MMEDIT, model, dataloader)
        assert outputs is not None
        api_test.post_process_outputs(outputs, dataset, model_cfg,
                                      Codebase.MMEDIT)
