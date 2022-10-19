# Copyright (c) OpenMMLab. All rights reserved.
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest
import torch
from mmengine import Config

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper

try:
    import_codebase(Codebase.MMACTION)
except ImportError:
    pytest.skip(
        f'{Codebase.MMACTION} is not installed.', allow_module_level=True)

model_cfg_path = 'tests/test_codebase/test_mmaction/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmaction', task='VideoRecognition'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['output'])))

onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
img_shape = (224, 224)
num_classes = 400
video = 'tests/test_codebase/test_mmaction/data/video/demo.mp4'


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'output': torch.rand(1, num_classes),
    })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_build_backend_model(backend_model):
    assert isinstance(backend_model, torch.nn.Module)


def test_create_input():
    inputs = task_processor.create_input(video, input_shape=img_shape)
    assert isinstance(inputs, tuple) and len(inputs) == 2


def test_build_pytorch_model():
    from mmaction.models.recognizers.base import BaseRecognizer
    model = task_processor.build_pytorch_model(None)
    assert isinstance(model, BaseRecognizer)


def test_get_tensor_from_input():
    input_data = {'inputs': torch.ones(3, 4, 5)}
    inputs = task_processor.get_tensor_from_input(input_data)
    assert torch.equal(inputs, torch.ones(3, 4, 5))


def test_get_model_name():
    model_name = task_processor.get_model_name()
    assert isinstance(model_name, str) and model_name is not None


def test_build_dataset_and_dataloader():
    from torch.utils.data import DataLoader, Dataset
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg.test_dataloader.dataset)
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader_cfg = task_processor.model_cfg.test_dataloader
    dataloader = task_processor.build_dataloader(dataloader_cfg)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


def test_build_test_runner(backend_model):
    from mmdeploy.codebase.base.runner import DeployTestRunner
    temp_dir = TemporaryDirectory().name
    runner = task_processor.build_test_runner(backend_model, temp_dir)
    assert isinstance(runner, DeployTestRunner)


def test_get_preprocess():
    process = task_processor.get_preprocess()
    assert process is not None


def test_get_postprocess():
    process = task_processor.get_postprocess()
    assert isinstance(process, dict)
