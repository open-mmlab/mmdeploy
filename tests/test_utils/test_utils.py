import os
import tempfile

import mmcv
import pytest

from mmdeploy.utils import get_onnx_config, get_task_type, load_config
from mmdeploy.utils.constants import Task
from mmdeploy.utils.export_info import dump_info

correct_model_path = 'tests/data/srgan.py'
correct_model_cfg = mmcv.Config.fromfile(correct_model_path)
correct_deploy_path = 'tests/data/super-resolution.py'
correct_deploy_cfg = mmcv.Config.fromfile(correct_deploy_path)
empty_file_path = tempfile.NamedTemporaryFile(suffix='.py').name
empty_path = './a.py'


@pytest.fixture(autouse=True, scope='module')
def create_empty_file():
    os.mknod(empty_file_path)


def test_load_config_none():
    with pytest.raises(AssertionError):
        load_config()


def test_load_config_type_error():
    with pytest.raises(TypeError):
        load_config(1)


def test_load_config_file_error():
    with pytest.raises(FileNotFoundError):
        load_config(empty_path)


@pytest.mark.parametrize('args', [
    [empty_file_path],
    [correct_model_path],
    [correct_model_cfg],
    (correct_model_path, correct_deploy_path),
    (correct_model_path, correct_deploy_cfg),
    (correct_model_cfg, correct_deploy_cfg),
])
def test_load_config(args):
    configs = load_config(*args)
    for v in zip(configs, args):
        if isinstance(v[1], str):
            cfg = mmcv.Config.fromfile(v[1])
        else:
            cfg = v[1]
        assert v[0]._cfg_dict == cfg._cfg_dict


@pytest.mark.parametrize('deploy_cfg, default',
                         [(empty_file_path, None),
                          (empty_file_path, Task.SUPER_RESOLUTION)])
def test_get_task_type_default(deploy_cfg, default):
    if default is None:
        res = get_task_type(deploy_cfg)
    else:
        res = get_task_type(deploy_cfg, default)
    assert res == default


@pytest.mark.parametrize('deploy_cfg, default',
                         [(correct_deploy_path, None),
                          (correct_deploy_path, Task.TEXT_DETECTION),
                          (correct_deploy_cfg, None)])
def test_get_task_type(deploy_cfg, default):
    if default is None:
        res = get_task_type(deploy_cfg)
    else:
        res = get_task_type(deploy_cfg, default)
    assert res == Task.SUPER_RESOLUTION


def test_get_onnx_config_error():
    with pytest.raises(Exception):
        get_onnx_config(empty_file_path)


@pytest.mark.parametrize('deploy_cfg',
                         [correct_deploy_path, correct_deploy_cfg])
def test_get_onnx_config(deploy_cfg):
    onnx_config = dict(
        dynamic_axes={
            'input': {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'output': {
                0: 'batch',
                2: 'height',
                3: 'width'
            }
        },
        type='onnx',
        export_params=True,
        keep_initializers_as_inputs=False,
        opset_version=11,
        save_file='end2end.onnx',
        input_names=['input'],
        output_names=['output'],
        input_shape=None)
    res = get_onnx_config(deploy_cfg)
    assert res == onnx_config


def test_AdvancedEnum():
    keys = [
        Task.TEXT_DETECTION, Task.TEXT_RECOGNITION, Task.SEGMENTATION,
        Task.SUPER_RESOLUTION, Task.CLASSIFICATION, Task.OBJECT_DETECTION
    ]
    vals = [
        'TextDetection', 'TextRecognition', 'Segmentation', 'SuperResolution',
        'Classification', 'ObjectDetection'
    ]
    for k, v in zip(keys, vals):
        assert Task.get(v, None) == k
        assert k.value == v
    assert Task.get('a', Task.TEXT_DETECTION) == Task.TEXT_DETECTION


def test_export_info():
    with tempfile.TemporaryDirectory() as dir:
        dump_info(correct_deploy_cfg, correct_model_cfg, dir)
        preprocess_json = os.path.join(dir, 'preprocess.json')
        deploy_json = os.path.join(dir, 'deploy_cfg.json')
        assert os.path.exists(preprocess_json)
        assert os.path.exists(deploy_json)
