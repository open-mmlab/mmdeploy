import mmcv
import numpy as np
import torch
import torch.nn as nn

from mmdeploy.apis.utils import build_dataloader, build_dataset, create_input
from mmdeploy.mmseg.export import convert_syncbatchnorm
from mmdeploy.utils.constants import Codebase, Task


def test_convert_syncbatchnorm():

    class ExampleModel(nn.Module):

        def __init__(self):
            super(ExampleModel, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(2, 4), nn.SyncBatchNorm(4), nn.Sigmoid(),
                nn.Linear(4, 6), nn.SyncBatchNorm(6), nn.Sigmoid())

        def forward(self, x):
            return self.model(x)

    model = ExampleModel()
    out_model = convert_syncbatchnorm(model)
    assert isinstance(out_model.model[1],
                      torch.nn.modules.batchnorm.BatchNorm2d) and isinstance(
                          out_model.model[4],
                          torch.nn.modules.batchnorm.BatchNorm2d)


class TestCreateInput:
    task = Task.SEGMENTATION
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    img_test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(50, 50),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    imgs = np.random.rand(32, 32, 3)
    img_path = 'tests/test_mmseg/data/imgs/blank.jpg'

    def test_create_input_static(self):
        data = dict(test=dict(pipeline=TestCreateInput.img_test_pipeline))
        model_cfg = mmcv.Config(
            dict(data=data, test_pipeline=TestCreateInput.img_test_pipeline))
        inputs = create_input(
            Codebase.MMSEG,
            TestCreateInput.task,
            model_cfg,
            TestCreateInput.imgs,
            input_shape=(32, 32),
            device='cpu')
        assert inputs is not None, 'Failed to create input'

    def test_create_input_dynamic(self):
        data = dict(test=dict(pipeline=TestCreateInput.img_test_pipeline))
        model_cfg = mmcv.Config(
            dict(data=data, test_pipeline=TestCreateInput.img_test_pipeline))
        inputs = create_input(
            Codebase.MMSEG,
            TestCreateInput.task,
            model_cfg,
            TestCreateInput.imgs,
            input_shape=None,
            device='cpu')
        assert inputs is not None, 'Failed to create input'

    def test_create_input_from_file(self):
        data = dict(test=dict(pipeline=TestCreateInput.img_test_pipeline))
        model_cfg = mmcv.Config(
            dict(data=data, test_pipeline=TestCreateInput.img_test_pipeline))
        inputs = create_input(
            Codebase.MMSEG,
            TestCreateInput.task,
            model_cfg,
            TestCreateInput.img_path,
            input_shape=None,
            device='cpu')
        assert inputs is not None, 'Failed to create input'


def test_build_dataset():
    data = dict(
        test={
            'type': 'CityscapesDataset',
            'data_root': 'tests/data',
            'img_dir': '',
            'ann_dir': '',
            'pipeline': [
                {
                    'type': 'LoadImageFromFile'
                },
            ]
        })
    dataset_cfg = mmcv.Config(dict(data=data))
    dataset = build_dataset(
        Codebase.MMSEG, dataset_cfg=dataset_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = build_dataloader(Codebase.MMSEG, dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'
