import mmcv
import numpy as np

from mmdeploy.apis.utils import build_dataloader, build_dataset, create_input
from mmdeploy.utils.constants import Codebase, Task


class TestCreateInput:
    task = Task.SUPER_RESOLUTION
    img_test_pipeline = [
        dict(
            type='LoadImageFromFile',
            io_backend='disk',
            key='lq',
            flag='unchanged'),
        dict(
            type='LoadImageFromFile',
            io_backend='disk',
            key='gt',
            flag='unchanged'),
        dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
        dict(
            type='Normalize',
            keys=['lq', 'gt'],
            mean=[0, 0, 0],
            std=[1, 1, 1],
            to_rgb=True),
        dict(
            type='Collect',
            keys=['lq', 'gt'],
            meta_keys=['lq_path', 'lq_path']),
        dict(type='ImageToTensor', keys=['lq', 'gt'])
    ]

    imgs = np.random.rand(32, 32, 3)
    img_path = 'tests/test_mmedit/data/imgs/blank.jpg'

    def test_create_input_static(this):
        data = dict(test=dict(pipeline=TestCreateInput.img_test_pipeline))
        model_cfg = mmcv.Config(
            dict(data=data, test_pipeline=TestCreateInput.img_test_pipeline))
        inputs = create_input(
            Codebase.MMEDIT,
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
            Codebase.MMEDIT,
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
            Codebase.MMEDIT,
            TestCreateInput.task,
            model_cfg,
            TestCreateInput.img_path,
            input_shape=None,
            device='cpu')
        assert inputs is not None, 'Failed to create input'


def test_build_dataset():
    data = dict(
        test={
            'type': 'SRFolderDataset',
            'lq_folder': 'tests/test_mmedit/data/imgs',
            'gt_folder': 'tests/test_mmedit/data/imgs',
            'scale': 1,
            'filename_tmpl': '{}',
            'pipeline': [
                {
                    'type': 'LoadImageFromFile'
                },
            ]
        })
    dataset_cfg = mmcv.Config(dict(data=data))
    dataset = build_dataset(
        Codebase.MMEDIT, dataset_cfg=dataset_cfg, dataset_type='test')
    assert dataset is not None, 'Failed to build dataset'
    dataloader = build_dataloader(Codebase.MMEDIT, dataset, 1, 1)
    assert dataloader is not None, 'Failed to build dataloader'
