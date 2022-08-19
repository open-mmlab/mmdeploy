# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import tempfile

import mmcv
import pytest

from mmdeploy.apis import torch2torchscript
from mmdeploy.utils import IR, Backend
from mmdeploy.utils.test import get_random_name

ts_file = tempfile.NamedTemporaryFile(suffix='.pt').name
input_name = get_random_name()
output_name = get_random_name()


def get_deploy_cfg(input_name, output_name):
    return mmcv.Config(
        dict(
            ir_config=dict(
                type=IR.TORCHSCRIPT.value,
                input_names=[input_name],
                output_names=[output_name],
                input_shape=None),
            codebase_config=dict(type='mmedit', task='SuperResolution'),
            backend_config=dict(type=Backend.TORCHSCRIPT.value)))


def get_model_cfg():
    return mmcv.Config(
        dict(
            model=dict(
                pretrained=None,
                type='BasicRestorer',
                generator=dict(
                    type='RRDBNet',
                    in_channels=3,
                    out_channels=3,
                    mid_channels=64,
                    num_blocks=23,
                    growth_channels=32),
                pixel_loss=dict(
                    type='L1Loss', loss_weight=1.0, reduction='mean')),
            test_cfg=dict(metrics='PSNR'),
            test_pipeline=[
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
            ]))


@pytest.mark.parametrize('input_name', [input_name])
@pytest.mark.parametrize('output_name', [output_name])
@pytest.mark.skipif(
    not importlib.util.find_spec('mmedit'), reason='requires mmedit')
def test_torch2torchscript(input_name, output_name):
    import numpy as np
    deploy_cfg = get_deploy_cfg(input_name, output_name)
    torch2torchscript(
        np.random.rand(8, 8, 3),
        '',
        ts_file,
        deploy_cfg,
        model_cfg=get_model_cfg(),
        device='cpu')

    print(ts_file)
    assert osp.exists(ts_file)
