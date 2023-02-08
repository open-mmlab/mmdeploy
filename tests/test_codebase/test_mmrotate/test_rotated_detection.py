# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import load_config
from mmdeploy.utils.test import DummyModel, SwitchBackendWrapper

model_cfg_path = 'tests/test_codebase/test_mmrotate/data/model.py'


@pytest.fixture(scope='module')
def model_cfg():
    return load_config(model_cfg_path)[0]


@pytest.fixture(scope='module')
def deploy_cfg():
    return mmcv.Config(
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


@pytest.fixture(scope='module')
def task_processor(model_cfg, deploy_cfg):
    return build_task_processor(model_cfg, deploy_cfg, 'cpu')


img_shape = (32, 32)


@pytest.fixture(scope='module')
def img():
    return np.random.rand(*img_shape, 3)


@pytest.fixture(scope='module')
def torch_model(task_processor):
    return task_processor.init_pytorch_model(None)


def test_init_pytorch_model(torch_model):
    from mmrotate.models import RotatedBaseDetector
    assert isinstance(torch_model, RotatedBaseDetector)


@pytest.fixture(scope='module')
def backend_model(task_processor):
    from mmdeploy.backend.onnxruntime import ORTWrapper
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs={
            'dets': torch.rand(1, 10, 6),
            'labels': torch.rand(1, 10)
        })

        yield task_processor.init_backend_model([''])


def test_init_backend_model(backend_model):
    from mmdeploy.codebase.mmrotate.deploy.rotated_detection_model import \
        End2EndModel
    assert isinstance(backend_model, End2EndModel)


@pytest.fixture(scope='module')
def model_inputs(task_processor, img):
    return task_processor.create_input(img, input_shape=img_shape)


@pytest.mark.parametrize('device', ['cpu'])
def test_create_input(device, task_processor, model_inputs):
    original_device = task_processor.device
    task_processor.device = device
    assert len(model_inputs) == 2
    task_processor.device = original_device


def test_run_inference(backend_model, task_processor, torch_model,
                       model_inputs):
    input_dict, _ = model_inputs
    torch_results = task_processor.run_inference(torch_model, input_dict)
    backend_results = task_processor.run_inference(backend_model, input_dict)
    assert torch_results is not None
    assert backend_results is not None
    assert len(torch_results[0]) == len(backend_results[0])


def test_visualize(backend_model, task_processor, model_inputs, img, tmp_path):
    input_dict, _ = model_inputs
    results = task_processor.run_inference(backend_model, input_dict)
    filename = str(tmp_path / 'tmp.jpg')
    task_processor.visualize(backend_model, img, results[0], filename, '')
    assert os.path.exists(filename)


def test_get_partition_cfg(task_processor):
    with pytest.raises(NotImplementedError):
        _ = task_processor.get_partition_cfg(partition_type='')


def test_build_dataset_and_dataloader(task_processor, model_cfg):
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg, dataset_type='test')
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


def test_single_gpu_test_and_evaluate(task_processor, model_cfg, tmp_path):
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
    output_file = str(tmp_path / 'tmp.pkl')
    task_processor.evaluate_outputs(
        model_cfg, outputs, dataset, 'bbox', out=output_file, format_only=True)
