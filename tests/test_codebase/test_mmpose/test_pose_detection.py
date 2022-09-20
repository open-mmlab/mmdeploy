# Copyright (c) OpenMMLab. All rights reserved.
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import SwitchBackendWrapper

try:
    import_codebase(Codebase.MMPOSE)
except ImportError:
    pytest.skip(
        f'{Codebase.MMPOSE.value} is not installed.', allow_module_level=True)

from .utils import (generate_datasample, generate_mmpose_deploy_config,
                    generate_mmpose_task_processor)

model_cfg_path = 'tests/test_codebase/test_mmpose/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = generate_mmpose_deploy_config()

onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = generate_mmpose_task_processor()
img_shape = (192, 256)
heatmap_shape = (48, 64)
# mmpose.apis.inference.LoadImage uses opencv, needs float32 in
# cv2.cvtColor.
img = np.random.rand(*img_shape, 3).astype(np.float32)
img_path = 'tests/data/tiger.jpeg'
num_output_channels = 17


@pytest.mark.parametrize('imgs', [img, img_path])
def test_create_input(imgs):
    inputs = task_processor.create_input(imgs, input_shape=img_shape)
    assert isinstance(inputs, tuple) and len(inputs) == 2


def test_build_pytorch_model():
    from mmpose.models.pose_estimators.base import BasePoseEstimator
    model = task_processor.build_pytorch_model(None)
    assert isinstance(model, BasePoseEstimator)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'output': torch.rand(1, num_output_channels, *heatmap_shape),
    })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_build_backend_model(backend_model):
    assert isinstance(backend_model, torch.nn.Module)


def test_visualize():
    datasample = generate_datasample(img.shape[:2])
    output_file = NamedTemporaryFile(suffix='.jpg').name
    task_processor.visualize(
        img, datasample, output_file, show_result=False, window_name='test')


def test_get_tensor_from_input():
    data = torch.ones(3, 4, 5)
    input_data = {'inputs': data}
    inputs = task_processor.get_tensor_from_input(input_data)
    assert torch.equal(inputs, data)


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
    val_dataloader = model_cfg['val_dataloader']
    dataset = task_processor.build_dataset(
        dataset_cfg=val_dataloader['dataset'])
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(val_dataloader)
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
