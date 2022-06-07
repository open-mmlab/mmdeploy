# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend, Codebase, Task, load_config
from mmdeploy.utils.test import DummyModel, SwitchBackendWrapper

try:
    import_codebase(Codebase.MMPOSE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMPOSE.value} is not installed.', allow_module_level=True)

model_cfg_path = 'tests/test_codebase/test_mmpose/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = mmcv.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmpose', task='PoseDetection'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            save_file='end2end.onnx',
            input_names=['input'],
            output_names=['output'],
            input_shape=None)))

onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
img_shape = (192, 256)
heatmap_shape = (48, 64)
# mmpose.apis.inference.LoadImage uses opencv, needs float32 in
# cv2.cvtColor.
img = np.random.rand(*img_shape, 3).astype(np.float32)
num_output_channels = model_cfg['data_cfg']['num_output_channels']


def test_create_input():
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(type=Backend.ONNXRUNTIME.value),
            codebase_config=dict(
                type=Codebase.MMPOSE.value, task=Task.POSE_DETECTION.value),
            onnx_config=dict(
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                save_file='end2end.onnx',
                input_names=['input'],
                output_names=['output'],
                input_shape=None)))
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    inputs = task_processor.create_input(img, input_shape=img_shape)
    assert isinstance(inputs, tuple) and len(inputs) == 2


def test_init_pytorch_model():
    from mmpose.models.detectors.base import BasePose
    model = task_processor.init_pytorch_model(None)
    assert isinstance(model, BasePose)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'output': torch.rand(1, num_output_channels, *heatmap_shape),
    })

    yield task_processor.init_backend_model([''])

    wrapper.recover()


def test_init_backend_model(backend_model):
    assert isinstance(backend_model, torch.nn.Module)


def test_run_inference(backend_model):
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    results = task_processor.run_inference(backend_model, input_dict)
    assert results is not None


def test_visualize(backend_model):
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    results = task_processor.run_inference(backend_model, input_dict)
    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(backend_model, img, results[0], filename, '')
        assert os.path.exists(filename)


def test_get_tensor_from_input():
    input_data = {'img': torch.ones(3, 4, 5)}
    inputs = task_processor.get_tensor_from_input(input_data)
    assert torch.equal(inputs, torch.ones(3, 4, 5))


def test_get_partition_cfg():
    try:
        _ = task_processor.get_partition_cfg(partition_type='')
    except NotImplementedError:
        pass


def test_get_model_name():
    model_name = task_processor.get_model_name()
    assert isinstance(model_name, str) and model_name is not None


def test_build_dataset_and_dataloader():
    from torch.utils.data import DataLoader, Dataset
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg, dataset_type='test')
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


def test_single_gpu_test_and_evaluate():
    from mmcv.parallel import MMDataParallel
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg, dataset_type='test')
    dataloader = task_processor.build_dataloader(dataset, 1, 1)

    # Prepare dummy model
    model = DummyModel(outputs=[torch.rand([1, 1000])])
    model = MMDataParallel(model, device_ids=[0])
    assert model is not None
    # Run test
    outputs = task_processor.single_gpu_test(model, dataloader)
    assert outputs is not None
    task_processor.evaluate_outputs(model_cfg, outputs, dataset)
