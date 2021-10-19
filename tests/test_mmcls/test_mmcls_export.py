import mmcv
import numpy as np

from mmdeploy.apis.utils import build_dataloader, build_dataset, create_input
from mmdeploy.utils.constants import Codebase, Task


class TestCreateInput:
    task = Task.CLASSIFICATION
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    img_test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=(256, -1)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ]

    imgs = np.random.rand(32, 32, 3)
    img_path = 'tests/test_mmcls/data/imgs/blank.jpg'

    def test_create_input_static(this):
        data = dict(test=dict(pipeline=TestCreateInput.img_test_pipeline))
        model_cfg = mmcv.Config(
            dict(data=data, test_pipeline=TestCreateInput.img_test_pipeline))
        inputs = create_input(
            Codebase.MMCLS,
            TestCreateInput.task,
            model_cfg,
            TestCreateInput.imgs,
            input_shape=(32, 32),
            device='cpu')
        assert inputs is not None, 'Failed to create input'

    def test_create_input_dynamic(this):
        data = dict(test=dict(pipeline=TestCreateInput.img_test_pipeline))
        model_cfg = mmcv.Config(
            dict(data=data, test_pipeline=TestCreateInput.img_test_pipeline))
        inputs = create_input(
            Codebase.MMCLS,
            TestCreateInput.task,
            model_cfg,
            TestCreateInput.imgs,
            input_shape=None,
            device='cpu')
        assert inputs is not None, 'Failed to create input'

    def test_create_input_from_file(this):
        data = dict(test=dict(pipeline=TestCreateInput.img_test_pipeline))
        model_cfg = mmcv.Config(
            dict(data=data, test_pipeline=TestCreateInput.img_test_pipeline))
        inputs = create_input(
            Codebase.MMCLS,
            TestCreateInput.task,
            model_cfg,
            TestCreateInput.img_path,
            input_shape=None,
            device='cpu')
        assert inputs is not None, 'Failed to create input'


def test_build_dataset():
    data = dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        test=dict(
            type='ImageNet',
            data_prefix='tests/test_mmcls/data/imgs',
            ann_file='tests/test_mmcls/data/imgs/ann.txt',
            pipeline=[
                {
                    'type': 'LoadImageFromFile'
                },
            ]))
    dataset_cfg = mmcv.Config(dict(data=data))
    dataset = build_dataset(
        Codebase.MMCLS, dataset_cfg=dataset_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = build_dataloader(Codebase.MMCLS, dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'
