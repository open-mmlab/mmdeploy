# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import torch
from mmengine import Config
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper

try:
    import_codebase(Codebase.MMROTATE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMROTATE} is not installed.', allow_module_level=True)

model_cfg_path = 'tests/test_codebase/test_mmrotate/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = Config(
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
task_processor = None
img_shape = (32, 32)
img = np.random.rand(*img_shape, 3)


@pytest.fixture(autouse=True)
def init_task_processor():
    global task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


def test_build_pytorch_model():
    from mmdet.models import BaseDetector
    model = task_processor.build_pytorch_model(None)
    assert isinstance(model, BaseDetector)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'dets': torch.rand(1, 10, 6),
        'labels': torch.randint(1, 10, (1, 10))
    })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_build_backend_model(backend_model):
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


def test_visualize(backend_model):
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    results = backend_model.test_step(input_dict)[0]
    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(img, results, filename, 'window')
        assert os.path.exists(filename)


def test_get_partition_cfg():
    with pytest.raises(NotImplementedError):
        _ = task_processor.get_partition_cfg(partition_type='')


def test_build_dataset_and_dataloader():
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg.test_dataloader.dataset)
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader_cfg = task_processor.model_cfg.test_dataloader
    dataloader = task_processor.build_dataloader(dataloader_cfg)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'
