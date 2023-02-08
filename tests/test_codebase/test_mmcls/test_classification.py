# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Any

import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.apis import build_task_processor
from mmdeploy.utils import load_config
from mmdeploy.utils.test import DummyModel, SwitchBackendWrapper

model_cfg_path = 'tests/test_codebase/test_mmcls/data/model.py'


@pytest.fixture(scope='module')
def model_cfg():
    return load_config(model_cfg_path)[0]


@pytest.fixture(scope='module')
def deploy_cfg():
    return mmcv.Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(type='mmcls', task='Classification'),
            onnx_config=dict(
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                input_shape=None,
                input_names=['input'],
                output_names=['output'])))


img_shape = (64, 64)
num_classes = 1000


@pytest.fixture(scope='module')
def task_processor(model_cfg, deploy_cfg):
    return build_task_processor(model_cfg, deploy_cfg, 'cpu')


@pytest.fixture(scope='module')
def img():
    return np.random.rand(*img_shape, 3)


@pytest.mark.parametrize('from_mmrazor', [True, False, '123', 0])
def test_init_pytorch_model(from_mmrazor: Any, task_processor, deploy_cfg):
    from mmcls.models.classifiers.base import BaseClassifier
    if from_mmrazor is False:
        _task_processor = task_processor
    else:
        _model_cfg_path = 'tests/test_codebase/test_mmcls/data/' \
            'mmrazor_model.py'
        _model_cfg = load_config(_model_cfg_path)[0]
        _model_cfg.algorithm.architecture.model.type = 'mmcls.ImageClassifier'
        _model_cfg.algorithm.architecture.model.backbone = dict(
            type='SearchableShuffleNetV2', widen_factor=1.0)
        _deploy_cfg = copy.deepcopy(deploy_cfg)
        _deploy_cfg.codebase_config['from_mmrazor'] = from_mmrazor
        _task_processor = build_task_processor(_model_cfg, _deploy_cfg, 'cpu')

    if not isinstance(from_mmrazor, bool):
        with pytest.raises(
                TypeError,
                match='`from_mmrazor` attribute must be '
                'boolean type! '
                f'but got: {from_mmrazor}'):
            _ = _task_processor.from_mmrazor
        return
    assert from_mmrazor == _task_processor.from_mmrazor
    if from_mmrazor:
        pytest.importorskip('mmrazor', reason='mmrazor is not installed.')
    model = _task_processor.init_pytorch_model(None)
    assert isinstance(model, BaseClassifier)


@pytest.fixture(scope='module')
def backend_model(task_processor):
    from mmdeploy.backend.onnxruntime import ORTWrapper
    with SwitchBackendWrapper(ORTWrapper) as wrapper:
        wrapper.set(outputs={
            'output': torch.rand(1, num_classes),
        })

        yield task_processor.init_backend_model([''])


def test_init_backend_model(backend_model):
    assert isinstance(backend_model, torch.nn.Module)


@pytest.fixture(scope='module')
def model_inputs(task_processor, img):
    return task_processor.create_input(img, input_shape=img_shape)


def test_create_input(model_inputs):
    inputs = model_inputs
    assert isinstance(inputs, tuple) and len(inputs) == 2


def test_run_inference(task_processor, backend_model, model_inputs):
    input_dict, _ = model_inputs
    results = task_processor.run_inference(backend_model, input_dict)
    assert results is not None


def test_visualize(task_processor, backend_model, tmp_path, img, model_inputs):
    input_dict, _ = model_inputs
    results = task_processor.run_inference(backend_model, input_dict)
    filename = str(tmp_path / 'tmp.jpg')
    task_processor.visualize(backend_model, img, results[0], filename, '')
    assert os.path.exists(filename)


def test_get_tensor_from_input(task_processor):
    input_data = {'img': torch.ones(3, 4, 5)}
    inputs = task_processor.get_tensor_from_input(input_data)
    assert torch.equal(inputs, torch.ones(3, 4, 5))


def test_get_partition_cfg(task_processor):
    with pytest.raises(NotImplementedError):
        task_processor.get_partition_cfg(partition_type='')


def test_build_dataset_and_dataloader(task_processor, model_cfg):
    from torch.utils.data import DataLoader, Dataset
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg, dataset_type='test')
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


def test_single_gpu_test_and_evaluate(task_processor, model_cfg):
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
