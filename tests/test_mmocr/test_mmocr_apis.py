import importlib

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.apis.ncnn as ncnn_apis
import mmdeploy.apis.onnxruntime as ort_apis
import mmdeploy.apis.ppl as ppl_apis
import mmdeploy.apis.tensorrt as trt_apis
from mmdeploy.mmocr.apis.inference import get_classes_from_config
from mmdeploy.mmocr.apis.visualize import show_result
from mmdeploy.utils.test import SwitchBackendWrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_TensorRTDetector():
    # force add backend wrapper regardless of plugins
    # make sure TensorRTDetector can use TRTWrapper inside itself
    from mmdeploy.apis.tensorrt.tensorrt_utils import TRTWrapper
    trt_apis.__dict__.update({'TRTWrapper': TRTWrapper})

    outputs = {
        'output': torch.rand(1, 3, 64, 64).cuda(),
    }
    with SwitchBackendWrapper(TRTWrapper) as wrapper:
        wrapper.set(outputs=outputs)
        from mmdeploy.mmocr.apis.inference import TensorRTDetector
        model_config = mmcv.Config.fromfile(
            'tests/test_mmocr/data/config/dbnet.py')
        trt_detector = TensorRTDetector('', model_config, 0, False)
        # watch from config
        imgs = [torch.rand(1, 3, 64, 64).cuda()]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'pad_shape': [64, 64, 3],
            'scale_factor': [1., 1., 1., 1.],
        }]]

        results = trt_detector.forward_of_backend(imgs, img_metas)
        assert results is not None, 'failed to get output \
            using TensorRTDetector'


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_ONNXRuntimeDetector():
    # force add backend wrapper regardless of plugins
    # make sure ONNXRuntimeDetector can use ORTWrapper inside itself
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    outputs = (torch.rand(1, 3, 64, 64))
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs=outputs)
        from mmdeploy.mmocr.apis.inference import ONNXRuntimeDetector
        model_config = mmcv.Config.fromfile(
            'tests/test_mmocr/data/config/dbnet.py')
        ort_detector = ONNXRuntimeDetector('', model_config, 0, False)
        imgs = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'pad_shape': [64, 64, 3],
            'scale_factor': [1., 1., 1., 1.],
        }]]

        results = ort_detector.forward_of_backend(imgs, img_metas)
        assert results is not None, 'failed to get output using '\
            'ONNXRuntimeDetector'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('pyppl'), reason='requires pyppl')
def test_PPLDetector():
    # force add backend wrapper regardless of plugins
    # make sure PPLDetector can use PPLWrapper inside itself
    from mmdeploy.apis.ppl.ppl_utils import PPLWrapper
    ppl_apis.__dict__.update({'PPLWrapper': PPLWrapper})

    outputs = (torch.rand(1, 3, 64, 64))
    with SwitchBackendWrapper(PPLWrapper) as wrapper:
        wrapper.set(outputs=outputs)
        from mmdeploy.mmocr.apis.inference import PPLDetector
        model_config = mmcv.Config.fromfile(
            'tests/test_mmocr/data/config/dbnet.py')
        ppl_detector = PPLDetector('', model_config, 0, False)
        imgs = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'pad_shape': [64, 64, 3],
            'scale_factor': [1., 1., 1., 1.],
        }]]

        results = ppl_detector.forward_of_backend(imgs, img_metas)
        assert results is not None, 'failed to get output using PPLDetector'


@pytest.mark.skipif(
    not importlib.util.find_spec('ncnn'), reason='requires ncnn')
def test_NCNNDetector():
    # force add backend wrapper regardless of plugins
    # make sure NCNNDetector can use NCNNWrapper inside itself
    from mmdeploy.apis.ncnn.ncnn_utils import NCNNWrapper
    ncnn_apis.__dict__.update({'NCNNWrapper': NCNNWrapper})

    outputs = {'output': torch.rand(1, 3, 64, 64)}
    with SwitchBackendWrapper(NCNNWrapper) as wrapper:
        wrapper.set(outputs=outputs)
        from mmdeploy.mmocr.apis.inference import NCNNDetector
        model_config = mmcv.Config.fromfile(
            'tests/test_mmocr/data/config/dbnet.py')
        ncnn_detector = NCNNDetector(['', ''], model_config, 0, False)
        imgs = [torch.rand(1, 3, 64, 64)]
        img_metas = [[{
            'ori_shape': [64, 64, 3],
            'img_shape': [64, 64, 3],
            'pad_shape': [64, 64, 3],
            'scale_factor': [1., 1., 1., 1.],
        }]]

        results = ncnn_detector.forward_of_backend(imgs, img_metas)
        assert results is not None, 'failed to get output using NCNNDetector'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('tensorrt'), reason='requires tensorrt')
def test_TensorRTRecognizer():
    # force add backend wrapper regardless of plugins
    # make sure TensorRTRecognizer can use TRTWrapper inside itself
    from mmdeploy.apis.tensorrt.tensorrt_utils import TRTWrapper
    trt_apis.__dict__.update({'TRTWrapper': TRTWrapper})

    outputs = {
        'output': torch.rand(1, 9, 37).cuda(),
    }
    with SwitchBackendWrapper(TRTWrapper) as wrapper:
        wrapper.set(outputs=outputs)
        from mmdeploy.mmocr.apis.inference import TensorRTRecognizer
        model_config = mmcv.Config.fromfile(
            'tests/test_mmocr/data/config/crnn.py')
        trt_recognizer = TensorRTRecognizer('', model_config, 0, False)
        # watch from config
        imgs = [torch.rand(1, 1, 32, 32).cuda()]
        img_metas = [[{'resize_shape': [32, 32], 'valid_ratio': 1.0}]]

        results = trt_recognizer.forward_of_backend(imgs, img_metas)
        assert results is not None, 'failed to get output using \
            TensorRTRecognizer'


@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_ONNXRuntimeRecognizer():
    # force add backend wrapper regardless of plugins
    # make sure ONNXRuntimeRecognizer can use ORTWrapper inside itself
    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    outputs = (torch.rand(1, 9, 37))
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs=outputs)
        from mmdeploy.mmocr.apis.inference import ONNXRuntimeRecognizer
        model_config = mmcv.Config.fromfile(
            'tests/test_mmocr/data/config/crnn.py')
        ort_recognizer = ONNXRuntimeRecognizer('', model_config, 0, False)
        imgs = [torch.rand(1, 1, 32, 32).numpy()]
        img_metas = [[{'resize_shape': [32, 32], 'valid_ratio': 1.0}]]

        results = ort_recognizer.forward_of_backend(imgs, img_metas)
        assert results is not None, 'failed to get output using '\
            'ONNXRuntimeRecognizer'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
@pytest.mark.skipif(
    not importlib.util.find_spec('pyppl'), reason='requires pyppl')
def test_PPLRecognizer():
    # force add backend wrapper regardless of plugins
    # make sure PPLRecognizer can use PPLWrapper inside itself
    from mmdeploy.apis.ppl.ppl_utils import PPLWrapper
    ppl_apis.__dict__.update({'PPLWrapper': PPLWrapper})

    outputs = (torch.rand(1, 9, 37))
    with SwitchBackendWrapper(PPLWrapper) as wrapper:
        wrapper.set(outputs=outputs)
        from mmdeploy.mmocr.apis.inference import PPLRecognizer
        model_config = mmcv.Config.fromfile(
            'tests/test_mmocr/data/config/crnn.py')
        ppl_recognizer = PPLRecognizer('', model_config, 0, False)
        imgs = [torch.rand(1, 1, 32, 32)]
        img_metas = [[{'resize_shape': [32, 32], 'valid_ratio': 1.0}]]

        results = ppl_recognizer.forward_of_backend(imgs, img_metas)
        assert results is not None, 'failed to get output using PPLRecognizer'


@pytest.mark.skipif(
    not importlib.util.find_spec('ncnn'), reason='requires ncnn')
def test_NCNNRecognizer():
    # force add backend wrapper regardless of plugins
    # make sure NCNNPSSDetector can use NCNNWrapper inside itself
    from mmdeploy.apis.ncnn.ncnn_utils import NCNNWrapper
    ncnn_apis.__dict__.update({'NCNNWrapper': NCNNWrapper})

    outputs = {'output': torch.rand(1, 9, 37)}
    with SwitchBackendWrapper(NCNNWrapper) as wrapper:
        wrapper.set(outputs=outputs)
        from mmdeploy.mmocr.apis.inference import NCNNRecognizer
        model_config = mmcv.Config.fromfile(
            'tests/test_mmocr/data/config/crnn.py')
        ncnn_recognizer = NCNNRecognizer(['', ''], model_config, 0, False)
        imgs = [torch.rand(1, 1, 32, 32)]
        img_metas = [[{'resize_shape': [32, 32], 'valid_ratio': 1.0}]]

        results = ncnn_recognizer.forward_of_backend(imgs, img_metas)
        assert results is not None, 'failed to get output using NCNNRecognizer'


@pytest.mark.parametrize(
    'task, model',
    [('TextDetection', 'tests/test_mmocr/data/config/dbnet.py'),
     ('TextRecognition', 'tests/test_mmocr/data/config/crnn.py')])
@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_build_ocr_processor(task, model):
    model_cfg = mmcv.Config.fromfile(model)
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmocr', task=task)))

    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.apis.utils import init_backend_model
        result_model = init_backend_model([''], model_cfg, deploy_cfg, -1)
        assert result_model is not None


@pytest.mark.parametrize('model', ['tests/test_mmocr/data/config/dbnet.py'])
def test_get_classes_from_config(model):
    get_classes_from_config(model)


@pytest.mark.parametrize(
    'task, model',
    [('TextDetection', 'tests/test_mmocr/data/config/dbnet.py')])
@pytest.mark.skipif(
    not importlib.util.find_spec('onnxruntime'), reason='requires onnxruntime')
def test_show_result(task, model):
    model_cfg = mmcv.Config.fromfile(model)
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmocr', task=task)))

    from mmdeploy.apis.onnxruntime.onnxruntime_utils import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})

    # simplify backend inference
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(model_cfg=model_cfg, deploy_cfg=deploy_cfg)
        from mmdeploy.apis.utils import init_backend_model
        detector = init_backend_model([''], model_cfg, deploy_cfg, -1)
        img = np.random.random((64, 64, 3))
        result = {'boundary_result': [[1, 2, 3, 4, 5], [2, 2, 0, 4, 5]]}
        import os.path
        import tempfile

        from mmdeploy.utils.constants import Backend
        with tempfile.TemporaryDirectory() as dir:
            filename = dir + 'tmp.jpg'
            show_result(
                detector,
                img,
                result,
                filename,
                Backend.ONNXRUNTIME,
                show=False)
            assert os.path.exists(filename)
