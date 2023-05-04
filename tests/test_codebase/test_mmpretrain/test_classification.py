# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import numpy as np
import pytest
import torch
from mmengine import Config

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import DummyModel, SwitchBackendWrapper

try:
    import_codebase(Codebase.MMPRETRAIN)
except ImportError:
    pytest.skip(
        f'{Codebase.MMPRETRAIN} is not installed.', allow_module_level=True)

model_cfg_path = 'tests/test_codebase/test_mmpretrain/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmpretrain', task='Classification'),
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
img_shape = (64, 64)
num_classes = 1000
img = np.random.rand(*img_shape, 3)


@pytest.fixture(autouse=True)
def init_task_processor():
    global task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


@pytest.mark.parametrize('from_mmrazor', [True, False, '123', 0])
def test_build_pytorch_model(from_mmrazor: Any):
    from mmpretrain.models.classifiers.base import BaseClassifier
    if from_mmrazor is False:
        _task_processor = task_processor
    else:
        _model_cfg_path = 'tests/test_codebase/test_mmpretrain/data/' \
            'mmrazor_model.py'
        _model_cfg = load_config(_model_cfg_path)[0]
        _model_cfg.algorithm.architecture.model.type = 'mmpretrain.' \
                                                       'ImageClassifier'
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
    model = _task_processor.build_pytorch_model(None)
    assert isinstance(model, BaseClassifier)


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
    inputs = task_processor.create_input(img, input_shape=img_shape)
    assert isinstance(inputs, tuple) and len(inputs) == 2


def test_visualize(backend_model):
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    results = backend_model.test_step(input_dict)[0]
    with TemporaryDirectory() as dir:
        filename = dir + '/tmp.jpg'
        task_processor.visualize(img, results, filename, 'window')
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


def test_build_dataset_and_dataloader():
    from torch.utils.data import DataLoader, Dataset
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg.test_dataloader.dataset)
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader_cfg = task_processor.model_cfg.test_dataloader
    dataloader = task_processor.build_dataloader(dataloader_cfg)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


def test_build_test_runner():
    # Prepare dummy model
    from mmengine.structures import LabelData
    from mmpretrain.structures import DataSample
    label = LabelData(
        label=torch.tensor([0]),
        score=torch.rand(10),
        metainfo=dict(num_classes=10))
    outputs = [
        DataSample(
            pred_label=torch.tensor([0]),
            _pred_label=label,
            gt_label=torch.tensor([0]),
            _gt_label=label,
            metainfo=dict(
                img_shape=(224, 224),
                img_path='',
                ori_shape=(300, 400),
                scale_factor=(0.8525, 0.8533333333333334)))
    ]
    model = DummyModel(outputs=outputs)
    assert model is not None
    # Run test
    with TemporaryDirectory() as dir:
        runner = task_processor.build_test_runner(model, dir)
        runner.test()
