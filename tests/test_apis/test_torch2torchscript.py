# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import tempfile

import pytest
from mmengine import Config

from mmdeploy.apis import torch2torchscript
from mmdeploy.utils import IR, Backend
from mmdeploy.utils.test import get_random_name

ts_file = tempfile.NamedTemporaryFile(suffix='.pt').name
input_name = get_random_name()
output_name = get_random_name()


def get_deploy_cfg(input_name, output_name):
    return Config(
        dict(
            ir_config=dict(
                type=IR.TORCHSCRIPT.value,
                input_names=[input_name],
                output_names=[output_name],
                input_shape=None),
            codebase_config=dict(type='mmagic', task='SuperResolution'),
            backend_config=dict(type=Backend.TORCHSCRIPT.value)))


def get_model_cfg():
    import mmengine
    file = 'tests/test_codebase/test_mmagic/data/model.py'
    model_cfg = mmengine.Config.fromfile(file)
    return model_cfg


@pytest.mark.parametrize('input_name', [input_name])
@pytest.mark.parametrize('output_name', [output_name])
@pytest.mark.skipif(
    not importlib.util.find_spec('mmagic'), reason='requires mmagic')
def test_torch2torchscript(input_name, output_name):
    import numpy as np
    deploy_cfg = get_deploy_cfg(input_name, output_name)
    torch2torchscript(
        np.random.randint(0, 255, (8, 8, 3)),
        '',
        ts_file,
        deploy_cfg,
        model_cfg=get_model_cfg(),
        device='cpu')

    assert osp.exists(ts_file)
