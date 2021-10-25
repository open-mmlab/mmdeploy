import mmcv
import numpy as np
import pytest

from mmdeploy.apis.utils import build_dataloader, build_dataset, create_input
from mmdeploy.utils.constants import Codebase, Task


@pytest.mark.parametrize('task', [Task.TEXT_DETECTION, Task.TEXT_RECOGNITION])
def test_create_input(task):
    if task == Task.TEXT_DETECTION:
        test = dict(
            type='IcdarDataset',
            pipeline=[{
                'type': 'LoadImageFromFile',
                'color_type': 'color_ignore_orientation'
            }, {
                'type':
                'MultiScaleFlipAug',
                'img_scale': (128, 64),
                'flip':
                False,
                'transforms': [
                    {
                        'type': 'Resize',
                        'img_scale': (256, 128),
                        'keep_ratio': True
                    },
                    {
                        'type': 'Normalize',
                        'mean': [123.675, 116.28, 103.53],
                        'std': [58.395, 57.12, 57.375],
                        'to_rgb': True
                    },
                    {
                        'type': 'Pad',
                        'size_divisor': 32
                    },
                    {
                        'type': 'ImageToTensor',
                        'keys': ['img']
                    },
                    {
                        'type': 'Collect',
                        'keys': ['img']
                    },
                ]
            }])
        imgs = [np.random.rand(128, 64, 3).astype(np.uint8)]
    elif task == Task.TEXT_RECOGNITION:
        test = dict(
            type='UniformConcatDataset',
            pipeline=[
                {
                    'type': 'LoadImageFromFile',
                    'color_type': 'grayscale'
                },
                {
                    'type': 'ResizeOCR',
                    'height': 32,
                    'min_width': 32,
                    'max_width': None,
                    'keep_aspect_ratio': True
                },
                {
                    'type': 'Normalize',
                    'mean': [127],
                    'std': [127]
                },
                {
                    'type': 'DefaultFormatBundle'
                },
                {
                    'type': 'Collect',
                    'keys': ['img'],
                    'meta_keys': ['filename', 'resize_shape', 'valid_ratio']
                },
            ])
        imgs = [np.random.random((32, 32, 3)).astype(np.uint8)]
    data = dict(test=test)
    model_cfg = mmcv.Config(dict(data=data))
    inputs = create_input(
        Codebase.MMOCR,
        task,
        model_cfg,
        imgs,
        input_shape=imgs[0].shape[0:2],
        device='cpu')
    assert inputs is not None, 'Failed to create input'


@pytest.mark.parametrize('task', [Task.TEXT_DETECTION, Task.TEXT_RECOGNITION])
def test_build_dataset(task):
    import tempfile
    import os
    ann_file, ann_path = tempfile.mkstemp()
    if task == Task.TEXT_DETECTION:
        data = dict(
            test={
                'type': 'IcdarDataset',
                'ann_file':
                'tests/test_mmocr/data/icdar2015/instances_test.json',
                'img_prefix': 'tests/test_mmocr/data/icdar2015/imgs',
                'pipeline': [
                    {
                        'type': 'LoadImageFromFile'
                    },
                ]
            })
    elif task == Task.TEXT_RECOGNITION:
        data = dict(
            test={
                'type': 'OCRDataset',
                'ann_file': ann_path,
                'img_prefix': '',
                'loader': {
                    'type': 'HardDiskLoader',
                    'repeat': 1,
                    'parser': {
                        'type': 'LineStrParser',
                        'keys': ['filename', 'text'],
                        'keys_idx': [0, 1],
                        'separator': ' '
                    }
                },
                'pipeline': [],
                'test_mode': True
            })
    dataset_cfg = mmcv.Config(dict(data=data))
    dataset = build_dataset(
        Codebase.MMOCR, dataset_cfg=dataset_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = build_dataloader(Codebase.MMOCR, dataset, 1, 1)
    os.close(ann_file)
    assert dataloader is not None, 'Failed to build dataloader'
