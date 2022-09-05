# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

from mmengine import Config
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import mmdeploy.apis.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper

import_codebase(Codebase.MMEDIT)

model_cfg = 'tests/test_codebase/test_mmedit/data/model.py'
model_cfg = load_config(model_cfg)[0]
deploy_cfg = Config(
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
input_img = np.random.rand(32, 32, 3)
img_shape = [32, 32]
input = {'img': input_img}
onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'output': torch.rand(3, 50, 50),
    })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_build_backend_model(backend_model):
    assert backend_model is not None


def test_create_input():
    inputs = task_processor.create_input(input_img, input_shape=img_shape)
    assert inputs is not None


def test_visualize(backend_model):
    input_dict, _ = task_processor.create_input(input_img,
                                                input_shape=img_shape)
    results = backend_model.test_step(input_dict)[0]
    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(input_img, results, filename, 'window')
        assert os.path.exists(filename)


def test_get_tensor_from_input():
    assert type(task_processor.get_tensor_from_input(input)) is not dict


def test_get_partition_cfg():
    with pytest.raises(NotImplementedError):
        task_processor.get_partition_cfg(None)


def test_build_dataset():
    data = dict(
        test={
            'type': 'SRFolderDataset',
            'lq_folder': 'tests/test_codebase/test_mmedit/data/imgs',
            'gt_folder': 'tests/test_codebase/test_mmedit/data/imgs',
            'scale': 1,
            'filename_tmpl': '{}',
            'pipeline': [
                {
                    'type': 'LoadImageFromFile'
                },
            ]
        })
    dataset_cfg = Config(dict(data=data))
    dataset = task_processor.build_dataset(
        dataset_cfg=dataset_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'


def test_build_dataset_and_dataloader():
    data = dict(
        test={
            'type': 'SRFolderDataset',
            'lq_folder': 'tests/test_codebase/test_mmedit/data/imgs',
            'gt_folder': 'tests/test_codebase/test_mmedit/data/imgs',
            'scale': 1,
            'filename_tmpl': '{}',
            'pipeline': [
                {
                    'type': 'LoadImageFromFile'
                },
            ]
        })
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg.test_dataloader[-1].dataset)
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader_cfg = task_processor.model_cfg.test_dataloader[-1]
    dataloader = task_processor.build_dataloader(dataloader_cfg)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'
