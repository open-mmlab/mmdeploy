# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import mmengine
import numpy as np
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper

model_cfg_path = 'tests/test_codebase/test_mmocr/data/crnn.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = mmengine.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmocr', task='TextRecognition'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['output'])))

onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = None
img_shape = (32, 32)
img = np.random.rand(*img_shape, 3).astype(np.uint8)


@pytest.fixture(autouse=True)
def init_task_processor():
    try:
        import_codebase(Codebase.MMOCR)
    except ImportError:
        pytest.skip(
            f'{Codebase.MMOCR} is not installed.', allow_module_level=True)
    global task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


def test_build_pytorch_model():
    from mmocr.utils.setup_env import register_all_modules
    register_all_modules()
    from mmocr.models.textrecog.recognizers import BaseRecognizer
    model = task_processor.build_pytorch_model(None)
    assert isinstance(model, BaseRecognizer)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'output': torch.rand(1, 9, 37),
    })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_build_backend_model(backend_model):
    assert isinstance(backend_model, torch.nn.Module)


def test_create_input():
    inputs = task_processor.create_input(img, input_shape=img_shape)
    assert isinstance(inputs, tuple) and len(inputs) == 2


def test_visualize(backend_model):
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    results = backend_model.test_step(input_dict)[0]
    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(img, results, filename, 'tmp')
        assert os.path.exists(filename)


def test_get_tensor_from_input():
    input_data = {'inputs': torch.ones(3, 4, 5)}
    inputs = task_processor.get_tensor_from_input(input_data)
    assert torch.equal(inputs, torch.ones(3, 4, 5))


def test_get_partition_cfg():
    try:
        _ = task_processor.get_partition_cfg(partition_type='')
    except NotImplementedError:
        pass
