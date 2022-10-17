# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from tempfile import NamedTemporaryFile

import mmcv
import numpy as np
import pytest
import torch

import mmdeploy.apis.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper

try:
    import_codebase(Codebase.MMEDIT)
except ImportError:
    pytest.skip(
        f'{Codebase.MMEDIT} is not installed.', allow_module_level=True)

model_cfg = 'tests/test_codebase/test_mmedit/data/model.py'
model_cfg = load_config(model_cfg)[0]
deploy_cfg = mmcv.Config(
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
input = {'lq': input_img}
onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


def test_init_pytorch_model():
    torch_model = task_processor.init_pytorch_model(None)
    assert torch_model is not None


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'output': torch.rand(3, 50, 50),
    })

    yield task_processor.init_backend_model([''])

    wrapper.recover()


def test_init_backend_model(backend_model):
    assert backend_model is not None


def test_create_input():
    inputs = task_processor.create_input(input_img, img_shape=img_shape)
    assert inputs is not None


def test_visualize(backend_model):
    result = task_processor.run_inference(backend_model, input)
    with tempfile.TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(backend_model, input_img, result[0], filename,
                                 'onnxruntime')
        assert os.path.exists(filename)


def test_run_inference(backend_model):
    results = task_processor.run_inference(backend_model, input)
    assert results is not None


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
    dataset_cfg = mmcv.Config(dict(data=data))
    dataset = task_processor.build_dataset(
        dataset_cfg=dataset_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'


def test_single_gpu_test(backend_model):
    from mmcv.parallel import MMDataParallel
    dataset = task_processor.build_dataset(model_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'
    backend_model = MMDataParallel(backend_model, device_ids=[0])
    outputs = task_processor.single_gpu_test(backend_model, dataloader)
    assert outputs is not None, 'Failed to test model'
