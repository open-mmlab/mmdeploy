# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import mmcv
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
    import_codebase(Codebase.MMDET3D)
except ImportError:
    pytest.skip(
        f'{Codebase.MMDET3D} is not installed.', allow_module_level=True)

model_cfg_path = 'tests/test_codebase/test_mmdet3d/data/model_cfg.py'
pcd_path = 'tests/test_codebase/test_mmdet3d/data/kitti/kitti_000008.bin'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = mmcv.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(type='mmdet3d', task='VoxelDetection'),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['voxels', 'num_points', 'coors'],
            output_names=['bboxes', 'scores', 'labels'])))
onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


def test_init_pytorch_model():
    from mmdet3d.models import Base3DDetector
    model = task_processor.init_pytorch_model(None)
    assert isinstance(model, Base3DDetector)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(
        outputs={
            'bboxes': torch.rand(1, 50, 7),
            'scores': torch.rand(1, 50),
            'labels': torch.rand(1, 50)
        })

    yield task_processor.init_backend_model([''])

    wrapper.recover()


def test_init_backend_model(backend_model):
    from mmdeploy.codebase.mmdet3d.deploy.voxel_detection_model import \
        VoxelDetectionModel
    assert isinstance(backend_model, VoxelDetectionModel)


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
def test_run_inference(backend_model):
    task_processor.device = 'cuda:0'
    torch_model = task_processor.init_pytorch_model(None)
    input_dict, _ = task_processor.create_input(pcd_path)
    torch_results = task_processor.run_inference(torch_model, input_dict)
    backend_results = task_processor.run_inference(backend_model, input_dict)
    assert torch_results is not None
    assert backend_results is not None
    assert len(torch_results[0]) == len(backend_results[0])
    task_processor.device = 'cpu'


@pytest.mark.skipif(
    reason='Only support GPU test', condition=not torch.cuda.is_available())
def test_visualize():
    task_processor.device = 'cuda:0'
    input_dict, _ = task_processor.create_input(pcd_path)
    torch_model = task_processor.init_pytorch_model(None)
    results = task_processor.run_inference(torch_model, input_dict)
    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.bin'
        task_processor.visualize(torch_model, pcd_path, results[0], filename,
                                 'test', False)
        assert os.path.exists(filename)
    task_processor.device = 'cpu'


def test_build_dataset_and_dataloader():
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg, dataset_type='test')
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


@pytest.mark.skipif(
    reason='Only support GPU test', condition=not torch.cuda.is_available())
def test_single_gpu_test_and_evaluate():
    from mmcv.parallel import MMDataParallel
    task_processor.device = 'cuda:0'

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
    model = DummyModel(outputs=[torch.rand([1, 10, 5]), torch.rand([1, 10])])
    model = MMDataParallel(model, device_ids=[0])
    # Run test
    outputs = task_processor.single_gpu_test(model, dataloader)
    assert isinstance(outputs, list)
    output_file = NamedTemporaryFile(suffix='.pkl').name
    task_processor.evaluate_outputs(
        model_cfg, outputs, dataset, 'bbox', out=output_file, format_only=True)
    task_processor.device = 'cpu'
