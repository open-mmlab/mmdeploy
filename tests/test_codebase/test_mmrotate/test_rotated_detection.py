# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import mmcv
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import DummyModel, SwitchBackendWrapper

try:
    import_codebase(Codebase.MMROTATE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMROTATE} is not installed.', allow_module_level=True)

model_cfg_path = 'tests/test_codebase/test_mmrotate/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = mmcv.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(
            type='mmrotate',
            task='RotatedDetection',
            post_processing=dict(
                score_threshold=0.05,
                iou_threshold=0.1,
                pre_top_k=2000,
                keep_top_k=2000)),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['dets', 'labels'])))
onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
img_shape = (32, 32)
img = np.random.rand(*img_shape, 3)


def test_init_pytorch_model():
    from mmrotate.models import RotatedBaseDetector
    model = task_processor.init_pytorch_model(None)
    assert isinstance(model, RotatedBaseDetector)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'dets': torch.rand(1, 10, 6),
        'labels': torch.rand(1, 10)
    })

    yield task_processor.init_backend_model([''])

    wrapper.recover()


def test_init_backend_model(backend_model):
    from mmdeploy.codebase.mmrotate.deploy.rotated_detection_model import \
        End2EndModel
    assert isinstance(backend_model, End2EndModel)


@pytest.mark.parametrize('device', ['cpu'])
def test_create_input(device):
    original_device = task_processor.device
    task_processor.device = device
    inputs = task_processor.create_input(img, input_shape=img_shape)
    assert len(inputs) == 2
    task_processor.device = original_device


def test_run_inference(backend_model):
    torch_model = task_processor.init_pytorch_model(None)
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    torch_results = task_processor.run_inference(torch_model, input_dict)
    backend_results = task_processor.run_inference(backend_model, input_dict)
    assert torch_results is not None
    assert backend_results is not None
    assert len(torch_results[0]) == len(backend_results[0])


def test_visualize(backend_model):
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    results = task_processor.run_inference(backend_model, input_dict)
    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(backend_model, img, results[0], filename, '')
        assert os.path.exists(filename)


def test_get_partition_cfg():
    with pytest.raises(NotImplementedError):
        _ = task_processor.get_partition_cfg(partition_type='')


def test_build_dataset_and_dataloader():
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg, dataset_type='test')
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


def test_single_gpu_test_and_evaluate():
    from mmcv.parallel import MMDataParallel

    class DummyDataset(Dataset):

        def __getitem__(self, index):
            return 0

        def __len__(self):
            return 0

        def evaluate(self, *args, **kwargs):
            return 0

        def format_results(self, *args, **kwargs):
            return 0

    dataset = DummyDataset()
    # Prepare dataloader
    dataloader = DataLoader(dataset)

    # Prepare dummy model
    model = DummyModel(outputs=[torch.rand([1, 10, 6]), torch.rand([1, 10])])
    model = MMDataParallel(model, device_ids=[0])
    # Run test
    outputs = task_processor.single_gpu_test(model, dataloader)
    assert isinstance(outputs, list)
    output_file = NamedTemporaryFile(suffix='.pkl').name
    task_processor.evaluate_outputs(
        model_cfg, outputs, dataset, 'bbox', out=output_file, format_only=True)
