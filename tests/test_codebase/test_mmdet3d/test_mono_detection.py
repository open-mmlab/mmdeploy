# Copyright (c) OpenMMLab. All rights reserved.
from tempfile import NamedTemporaryFile, TemporaryDirectory

import mmengine
import pytest
import torch

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Codebase, load_config
from mmdeploy.utils.test import DummyModel, SwitchBackendWrapper

try:
    import_codebase(Codebase.MMDET3D)
except ImportError:
    pytest.skip(
        f'{Codebase.MMDET3D} is not installed.', allow_module_level=True)
model_cfg_path = 'tests/test_codebase/test_mmdet3d/data/smoke_dla34_dlaneck_gn-all_4xb8-6x_kitti-mono3d.py'  # noqa: E501
pcd_path = 'tests/test_codebase/test_mmdet3d/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl'  # noqa: E501

model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = mmengine.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmdet3d', task='MonoDetection'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['cls_score', 'bbox_pred'])))
onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = None


@pytest.fixture(autouse=True)
def init_task_processor():
    global task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


def test_build_pytorch_model():
    from mmdet3d.models import SingleStageMono3DDetector
    model = task_processor.build_pytorch_model(None)
    assert isinstance(model, SingleStageMono3DDetector)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(
        outputs={
            'cls_score': torch.rand(1, 3, 96, 320),
            'bbox_pred': torch.rand(1, 8, 96, 320),
        })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_build_backend_model(backend_model):
    from mmdeploy.codebase.mmdet3d.deploy.mono_detection_model import \
        MonoDetectionModel
    assert isinstance(backend_model, MonoDetectionModel)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_create_input(device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('cuda is not available')
    original_device = task_processor.device
    task_processor.device = device
    inputs = task_processor.create_input(pcd_path)
    assert len(inputs) == 2
    task_processor.device = original_device


@pytest.mark.skipif(
    reason='Only support GPU test', condition=not torch.cuda.is_available())
def test_single_gpu_test_and_evaluate():
    task_processor.device = 'cuda:0'

    # Prepare dummy model
    model = DummyModel(
        outputs=[torch.rand([1, 3, 96, 320]),
                 torch.rand([1, 8, 96, 320])])

    assert model is not None
    # Run test
    with TemporaryDirectory() as dir:
        task_processor.build_test_runner(model, dir)
