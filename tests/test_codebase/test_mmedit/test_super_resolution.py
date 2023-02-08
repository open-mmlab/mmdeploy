# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import load_config
from mmdeploy.utils.test import SwitchBackendWrapper


@pytest.fixture(scope='module')
def model_cfg():
    cfg = 'tests/test_codebase/test_mmedit/data/model.py'
    return load_config(cfg)[0]


@pytest.fixture(scope='module')
def deploy_cfg():
    return mmcv.Config(
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


@pytest.fixture(scope='module')
def input_img():
    return np.random.rand(32, 32, 3)


@pytest.fixture(scope='module')
def model_input(input_img):
    return {'lq': input_img}


@pytest.fixture(scope='module')
def task_processor(model_cfg, deploy_cfg):
    return build_task_processor(model_cfg, deploy_cfg, 'cpu')


def test_init_pytorch_model(task_processor):
    torch_model = task_processor.init_pytorch_model(None)
    assert torch_model is not None


@pytest.fixture(scope='module')
def backend_model(task_processor):
    from mmdeploy.backend.onnxruntime import ORTWrapper
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs={
            'output': torch.rand(3, 50, 50),
        })

        yield task_processor.init_backend_model([''])


def test_init_backend_model(backend_model):
    assert backend_model is not None


def test_create_input(task_processor, input_img):
    inputs = task_processor.create_input(
        input_img, img_shape=input_img.shape[:2])
    assert inputs is not None


def test_visualize(backend_model, task_processor, model_input, input_img):
    result = task_processor.run_inference(backend_model, model_input)
    with tempfile.TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(backend_model, input_img, result[0], filename,
                                 'onnxruntime')
        assert os.path.exists(filename)


def test_run_inference(backend_model, task_processor, model_input):
    results = task_processor.run_inference(backend_model, model_input)
    assert results is not None


def test_get_tensor_from_input(task_processor, model_input):
    assert type(task_processor.get_tensor_from_input(model_input)) is not dict


def test_get_partition_cfg(task_processor):
    with pytest.raises(NotImplementedError):
        task_processor.get_partition_cfg(None)


def test_build_dataset(task_processor):
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
    dataset_cfg = mmcv.Config(dict(data=data))
    dataset = task_processor.build_dataset(
        dataset_cfg=dataset_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'


def test_single_gpu_test(backend_model, model_cfg, task_processor):
    from mmcv.parallel import MMDataParallel
    dataset = task_processor.build_dataset(model_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'
    backend_model = MMDataParallel(backend_model, device_ids=[0])
    outputs = task_processor.single_gpu_test(backend_model, dataloader)
    assert outputs is not None, 'Failed to test model'
