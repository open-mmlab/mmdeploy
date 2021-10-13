import mmcv
import numpy as np
import pytest
import torch

from mmdeploy.apis.utils import (build_dataloader, build_dataset, create_input,
                                 get_tensor_from_input)
from mmdeploy.utils.constants import Codebase, Task


def test_create_input():
    task = Task.OBJECT_DETECTION
    test = dict(pipeline=[{
        'type': 'LoadImageFromWebcam'
    }, {
        'type':
        'MultiScaleFlipAug',
        'img_scale': [32, 32],
        'flip':
        False,
        'transforms': [{
            'type': 'Resize',
            'keep_ratio': True
        }, {
            'type': 'RandomFlip'
        }, {
            'type': 'Normalize',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True
        }, {
            'type': 'Pad',
            'size_divisor': 32
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img']
        }]
    }])
    data = dict(test=test)
    model_cfg = mmcv.Config(dict(data=data))
    imgs = [np.random.rand(32, 32, 3)]
    inputs = create_input(
        Codebase.MMDET,
        task,
        model_cfg,
        imgs,
        input_shape=(32, 32),
        device='cpu')
    assert inputs is not None, 'Failed to create input'


@pytest.mark.parametrize('input_data', [{'img': [torch.ones(3, 4, 5)]}])
def test_get_tensor_from_input(input_data):
    inputs = get_tensor_from_input(Codebase.MMDET, input_data)
    assert inputs is not None, 'Failed to get tensor from input'


def test_build_dataset():
    data = dict(
        test={
            'type': 'CocoDataset',
            'ann_file': 'tests/test_mmdet/data/coco_sample.json',
            'img_prefix': 'tests/test_mmdet/data/imgs/',
            'pipeline': [
                {
                    'type': 'LoadImageFromFile'
                },
            ]
        })
    dataset_cfg = mmcv.Config(dict(data=data))
    dataset = build_dataset(
        Codebase.MMDET, dataset_cfg=dataset_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = build_dataloader(Codebase.MMDET, dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'


def test_clip_bboxes():
    from mmdeploy.mmdet.export import clip_bboxes
    x1 = torch.rand(3, 2) * 224
    y1 = torch.rand(3, 2) * 224
    x2 = x1 * 2
    y2 = y1 * 2
    outs = clip_bboxes(x1, y1, x2, y2, [224, 224])
    for out in outs:
        assert int(out.max()) <= 224


def test_pad_with_value():
    from mmdeploy.mmdet.export import pad_with_value
    x = torch.rand(3, 2)
    padded_x = pad_with_value(x, pad_dim=1, pad_size=4, pad_value=0)
    assert np.allclose(
        padded_x.shape, torch.Size([3, 6]), rtol=1e-03, atol=1e-05)
    assert np.allclose(padded_x.sum(), x.sum(), rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('partition_type', ['single_stage', 'two_stage'])
def test_get_partition_cfg(partition_type):
    from mmdeploy.mmdet.export import get_partition_cfg
    partition_cfg = get_partition_cfg(partition_type=partition_type)
    assert partition_cfg is not None
