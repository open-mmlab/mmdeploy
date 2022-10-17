# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import mmcv
import numpy as np
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
    import_codebase(Codebase.MMDET)
except ImportError:
    pytest.skip(f'{Codebase.MMDET} is not installed.', allow_module_level=True)

model_cfg_path = 'tests/test_codebase/test_mmdet/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
deploy_cfg = mmcv.Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(
            type='mmdet',
            task='ObjectDetection',
            post_processing=dict(
                score_threshold=0.05,
                confidence_threshold=0.005,  # for YOLOv3
                iou_threshold=0.5,
                max_output_boxes_per_class=200,
                pre_top_k=5000,
                keep_top_k=100,
                background_label_id=-1,
            )),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['dets', 'labels'])))
onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
img_shape = (32, 32)
img = np.random.rand(*img_shape, 3)


@pytest.mark.parametrize('from_mmrazor', [True, False, '123', 0])
def test_init_pytorch_model(from_mmrazor: Any):
    from mmdet.models import BaseDetector
    if from_mmrazor is False:
        _task_processor = task_processor
    else:
        _model_cfg_path = 'tests/test_codebase/test_mmdet/data/' \
            'mmrazor_model.py'
        _model_cfg = load_config(_model_cfg_path)[0]
        _model_cfg.algorithm.architecture.model.type = 'mmdet.YOLOV3'
        _model_cfg.algorithm.architecture.model.backbone.type = \
            'mmcls.SearchableShuffleNetV2'
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
    assert isinstance(model, BaseDetector)


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(outputs={
        'dets': torch.rand(1, 10, 5),
        'labels': torch.rand(1, 10)
    })

    yield task_processor.init_backend_model([''])

    wrapper.recover()


def test_init_backend_model(backend_model):
    from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
        End2EndModel
    assert isinstance(backend_model, End2EndModel)


def test_can_postprocess_masks():
    from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
        End2EndModel
    num_dets = [0, 1, 5]
    for num_det in num_dets:
        det_bboxes = np.random.randn(num_det, 4)
        det_masks = np.random.randn(num_det, 28, 28)
        img_w, img_h = (30, 40)
        masks = End2EndModel.postprocessing_masks(det_bboxes, det_masks, img_w,
                                                  img_h)
        expected_shape = (num_det, img_h, img_w)
        actual_shape = masks.shape
        assert actual_shape == expected_shape, \
            f'The expected shape of masks {expected_shape} ' \
            f'did not match actual shape {actual_shape}.'


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_create_input(device):
    if device == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('cuda is not available')
    original_device = task_processor.device
    task_processor.device = device
    inputs = task_processor.create_input(img, input_shape=img_shape)
    assert len(inputs) == 2
    task_processor.device = original_device


def test_run_inference(backend_model):
    torch_model = task_processor.init_pytorch_model(None)
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    torch_results = task_processor.run_inference(torch_model, input_dict)
    backend_results = task_processor.run_inference(backend_model, input_dict)
    assert torch_results is not None
    assert backend_results is not None
    assert len(torch_results[0]) == len(backend_results[0])


def test_visualize(backend_model):
    input_dict, _ = task_processor.create_input(img, input_shape=img_shape)
    results = task_processor.run_inference(backend_model, input_dict)
    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(backend_model, img, results[0], filename, '')
        assert os.path.exists(filename)


@pytest.mark.parametrize('partition_type', ['single_stage', 'two_stage'])
# Currently only mmdet implements get_partition_cfg
def test_get_partition_cfg(partition_type):
    from mmdeploy.codebase.mmdet.deploy.model_partition_cfg import \
        MMDET_PARTITION_CFG
    partition_cfg = task_processor.get_partition_cfg(
        partition_type=partition_type)
    assert partition_cfg == MMDET_PARTITION_CFG[partition_type]


def test_get_tensort_from_input():
    input_data = {'img': [torch.ones(3, 4, 5)]}
    inputs = task_processor.get_tensor_from_input(input_data)
    assert torch.equal(inputs, torch.ones(3, 4, 5))


def test_build_dataset_and_dataloader():
    dataset = task_processor.build_dataset(
        dataset_cfg=model_cfg, dataset_type='test')
    assert isinstance(dataset, Dataset), 'Failed to build dataset'
    dataloader = task_processor.build_dataloader(dataset, 1, 1)
    assert isinstance(dataloader, DataLoader), 'Failed to build dataloader'


def test_single_gpu_test_and_evaluate():
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
    model = DummyModel(outputs=[torch.rand([1, 10, 5]), torch.rand([1, 10])])
    model = MMDataParallel(model, device_ids=[0])
    # Run test
    outputs = task_processor.single_gpu_test(model, dataloader)
    assert isinstance(outputs, list)
    output_file = NamedTemporaryFile(suffix='.pkl').name
    task_processor.evaluate_outputs(
        model_cfg, outputs, dataset, 'bbox', out=output_file, format_only=True)
