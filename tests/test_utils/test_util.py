# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import logging
import os
import tempfile
from functools import partial

import pytest
import torch.multiprocessing as mp
from mmengine import Config

import mmdeploy.utils as util
from mmdeploy.backend.sdk.export_info import export2SDK
from mmdeploy.utils import target_wrapper
from mmdeploy.utils.config_utils import get_codebase_external_module
from mmdeploy.utils.constants import Backend, Codebase, Task
from mmdeploy.utils.test import get_random_name

correct_model_path = 'tests/test_codebase/test_mmagic/data/model.py'
correct_model_cfg = Config.fromfile(correct_model_path)
correct_deploy_path = 'tests/data/super-resolution.py'
correct_deploy_cfg = Config.fromfile(correct_deploy_path)
empty_file_path = tempfile.NamedTemporaryFile(suffix='.py').name
empty_path = './a.py'


@pytest.fixture(autouse=True, scope='module')
def create_empty_file():
    with open(empty_file_path, mode='w'):
        pass


class TestLoadConfigError:

    def test_load_config_none(self):
        with pytest.raises(AssertionError):
            util.load_config()

    def test_load_config_type_error(self):
        with pytest.raises(TypeError):
            util.load_config(1)

    def test_load_config_file_error(self):
        with pytest.raises(FileNotFoundError):
            util.load_config(empty_path)


class TestLoadConfig:

    @pytest.mark.parametrize('args', [
        [empty_file_path],
        [correct_model_path],
        [correct_model_cfg],
        (correct_model_path, correct_deploy_path),
        (correct_model_path, correct_deploy_cfg),
        (correct_model_cfg, correct_deploy_cfg),
    ])
    def test_load_config(self, args):
        configs = util.load_config(*args)
        for v in zip(configs, args):
            if isinstance(v[1], str):
                cfg = Config.fromfile(v[1])
            else:
                cfg = v[1]
            assert v[0]._cfg_dict == cfg._cfg_dict


class TestGetCodebaseConfig:

    def test_get_codebase_config_empty(self):
        assert util.get_codebase_config(Config(dict())) == {}

    def test_get_codebase_config(self):
        codebase_config = util.get_codebase_config(correct_deploy_path)
        assert isinstance(codebase_config, dict) and len(codebase_config) > 1


class TestGetTaskType:

    def test_get_task_type_none(self):
        with pytest.raises(AssertionError):
            util.get_task_type(Config(dict()))

    def test_get_task_type(self):
        assert util.get_task_type(correct_deploy_path) == Task.SUPER_RESOLUTION


class TestGetCodebase:

    def test_get_codebase_none(self):
        with pytest.raises(AssertionError):
            util.get_codebase(Config(dict()))

    def test_get_codebase(self):
        assert util.get_codebase(correct_deploy_path) == Codebase.MMAGIC


class TestGetBackendConfig:

    def test_get_backend_config_empty(self):
        assert util.get_backend_config(Config(dict())) == {}

    def test_get_backend_config(self):
        backend_config = util.get_backend_config(correct_deploy_path)
        assert isinstance(backend_config, dict) and len(backend_config) == 1


class TestGetCodebaseExternalModule:

    def test_get_codebase_external_module_empty(self):
        assert get_codebase_external_module(Config(dict())) == []

    def test_get_codebase_external_module(self):
        external_deploy_cfg = dict(
            onnx_config=dict(),
            codebase_config=dict(module=['mmyolo.deploy.mmyolo']),
            backend_config=dict(type='onnxruntime'))
        custom_module_list = get_codebase_external_module(external_deploy_cfg)
        assert isinstance(custom_module_list, list) \
            and len(custom_module_list) == 1


class TestGetBackend:

    def test_get_backend_none(self):
        with pytest.raises(AssertionError):
            util.get_backend(Config(dict()))

    def test_get_backend(self):
        assert util.get_backend(correct_deploy_path) == Backend.ONNXRUNTIME


class TestGetOnnxConfig:

    def test_get_onnx_config_empty(self):
        assert util.get_onnx_config(Config(dict())) == {}

    def test_get_onnx_config(self):
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
        assert util.get_onnx_config(correct_deploy_path) == onnx_config


class TestIsDynamic:

    config_with_onnx_config = Config(
        dict(onnx_config=dict(), backend_config=dict(type='default')))

    config_with_dynamic_axes = Config(
        dict(
            onnx_config=dict(
                type='onnx',
                dynamic_axes={'input': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'
                }}),
            backend_config=dict(type='default')))

    config_with_dynamic_axes_and_input_names = Config(
        dict(
            onnx_config=dict(
                type='onnx',
                input_names=['image'],
                dynamic_axes={'image': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'
                }}),
            backend_config=dict(type='default')))

    config_with_dynamic_axes_list = Config(
        dict(
            onnx_config=dict(
                type='onnx', input_names=['image'], dynamic_axes=[[0, 2, 3]]),
            backend_config=dict(type='default')))

    def test_is_dynamic_batch_none(self):
        assert util.is_dynamic_batch(
            TestIsDynamic.config_with_onnx_config) is False

    def test_is_dynamic_batch_error_name(self):
        assert util.is_dynamic_batch(TestIsDynamic.config_with_dynamic_axes,
                                     'output') is False

    def test_is_dynamic_batch(self):
        assert util.is_dynamic_batch(
            TestIsDynamic.config_with_dynamic_axes) is True

    def test_is_dynamic_batch_axes_list(self):
        assert util.is_dynamic_batch(
            TestIsDynamic.config_with_dynamic_axes_list) is True

    def test_is_dynamic_shape_none(self):
        assert util.is_dynamic_shape(
            TestIsDynamic.config_with_onnx_config) is False

    def test_is_dynamic_shape_error_name(self):
        assert util.is_dynamic_shape(TestIsDynamic.config_with_dynamic_axes,
                                     'output') is False

    def test_is_dynamic_shape(self):
        assert util.is_dynamic_shape(
            TestIsDynamic.config_with_dynamic_axes) is True

    def test_is_dynamic_shape_input_names(self):
        assert util.is_dynamic_shape(
            TestIsDynamic.config_with_dynamic_axes_and_input_names) is True

    def test_is_dynamic_shape_different_names(self):
        config_with_different_names = \
            TestIsDynamic.config_with_dynamic_axes_and_input_names
        util.get_ir_config(
            config_with_different_names).input_names = 'another_name'
        assert util.is_dynamic_shape(config_with_different_names) is False

    def test_is_dynamic_shape_axes_list(self):
        assert util.is_dynamic_shape(
            TestIsDynamic.config_with_dynamic_axes_list) is True


class TestGetInputShape:
    config_without_input_shape = Config(
        dict(onnx_config=dict(input_shape=None)))
    config_with_input_shape = Config(
        dict(onnx_config=dict(input_shape=[1, 1])))
    config_with_error_shape = Config(
        dict(onnx_config=dict(input_shape=[1, 1, 1])))

    def test_get_input_shape_none(self):
        assert util.get_input_shape(
            TestGetInputShape.config_without_input_shape) is None

    def test_get_input_shape_error(self):
        with pytest.raises(Exception):
            util.get_input_shape(TestGetInputShape.config_with_error_shape)

    def test_get_input_shape(self):
        assert util.get_input_shape(
            TestGetInputShape.config_with_input_shape) == [1, 1]


class TestCfgApplyMark:

    config_with_mask = Config(dict(partition_config=dict(apply_marks=True)))

    def test_cfg_apply_marks_none(self):
        assert util.cfg_apply_marks(Config(dict())) is None

    def test_cfg_apply_marks(self):
        assert util.cfg_apply_marks(TestCfgApplyMark.config_with_mask) is True


class TestGetPartitionConfig:

    config_with_mask = Config(dict(partition_config=dict(apply_marks=True)))
    config_without_mask = Config(
        dict(partition_config=dict(apply_marks=False)))

    def test_get_partition_config_none(self):
        assert util.get_partition_config(Config(dict())) is None

    def test_get_partition_config_without_mask(self):
        assert util.get_partition_config(
            TestGetPartitionConfig.config_without_mask) is None

    def test_get_partition_config(self):
        assert util.get_partition_config(
            TestGetPartitionConfig.config_with_mask) == dict(apply_marks=True)


class TestGetCalib:
    config_with_calib = Config(
        dict(calib_config=dict(create_calib=True, calib_file='calib_data.h5')))

    config_without_calib = Config(
        dict(
            calib_config=dict(create_calib=False, calib_file='calib_data.h5')))

    def test_get_calib_config(self):
        assert util.get_calib_config(TestGetCalib.config_with_calib) == dict(
            create_calib=True, calib_file='calib_data.h5')

    def test_get_calib_filename_none(self):
        assert util.get_calib_filename(Config(dict())) is None

    def test_get_calib_filename_false(self):
        assert util.get_calib_filename(
            TestGetCalib.config_without_calib) is None

    def test_get_calib_filename(self):
        assert util.get_calib_filename(
            TestGetCalib.config_with_calib) == 'calib_data.h5'


class TestGetCommonConfig:
    config_with_common_config = Config(
        dict(
            backend_config=dict(
                type='tensorrt', common_config=dict(fp16_mode=False))))

    def test_get_common_config(self):
        assert util.get_common_config(
            TestGetCommonConfig.config_with_common_config) == dict(
                fp16_mode=False)


class TestGetModelInputs:

    config_with_model_inputs = Config(
        dict(backend_config=dict(model_inputs=[dict(input_shapes=None)])))

    def test_model_inputs(self):
        assert util.get_model_inputs(
            TestGetModelInputs.config_with_model_inputs) == [
                dict(input_shapes=None)
            ]


class TestGetDynamicAxes:

    input_name = get_random_name()

    def test_with_empty_cfg(self):
        deploy_cfg = Config()
        with pytest.raises(KeyError):
            util.get_dynamic_axes(deploy_cfg)

    def test_can_get_axes_from_dict(self):
        expected_dynamic_axes = {
            self.input_name: {
                0: 'batch',
                2: 'height',
                3: 'width'
            }
        }
        deploy_cfg = Config(
            dict(onnx_config=dict(dynamic_axes=expected_dynamic_axes)))
        dynamic_axes = util.get_dynamic_axes(deploy_cfg)
        assert expected_dynamic_axes == dynamic_axes

    def test_can_not_get_axes_from_list_without_names(self):
        axes = [[0, 2, 3]]
        deploy_cfg = Config(dict(onnx_config=dict(dynamic_axes=axes)))
        with pytest.raises(KeyError):
            util.get_dynamic_axes(deploy_cfg)

    def test_can_get_axes_from_list_with_args(self):
        axes = [[0, 2, 3]]
        expected_dynamic_axes = {self.input_name: axes[0]}
        axes_names = [self.input_name]
        deploy_cfg = Config(dict(onnx_config=dict(dynamic_axes=axes)))
        dynamic_axes = util.get_dynamic_axes(deploy_cfg, axes_names)
        assert expected_dynamic_axes == dynamic_axes

    def test_can_get_axes_from_list_with_cfg(self):
        output_name = get_random_name()
        axes = [[0, 2, 3], [0]]
        expected_dynamic_axes = {
            self.input_name: axes[0],
            output_name: axes[1]
        }
        deploy_cfg = Config(
            dict(
                onnx_config=dict(
                    input_names=[self.input_name],
                    output_names=[output_name],
                    dynamic_axes=axes)))
        dynamic_axes = util.get_dynamic_axes(deploy_cfg)
        assert expected_dynamic_axes == dynamic_axes


class TestParseDeviceID:

    def test_cpu(self):
        device = 'cpu'
        assert util.parse_device_id(device) == -1
        assert util.parse_device_type(device) == 'cpu'

    def test_cuda(self):
        device = 'cuda'
        assert util.parse_device_id(device) == 0
        assert util.parse_device_type(device) == 'cuda'

    def test_cuda10(self):
        device = 'cuda:10'
        assert util.parse_device_id(device) == 10

    def test_incorrect_cuda_device(self):
        device = 'cuda_5'
        with pytest.raises(AssertionError):
            util.parse_device_id(device)

    def test_incorrect_device(self):
        device = 'abcd:1'
        assert util.parse_device_id(device) is None


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
        assert Task.get(v) == k
        assert k.value == v


@pytest.mark.skipif(
    not importlib.util.find_spec('mmagic'), reason='requires mmagic')
def test_export_info():
    with tempfile.TemporaryDirectory() as dir:
        export2SDK(correct_deploy_cfg, correct_model_cfg, dir, '', 'cpu')
        deploy_json = os.path.join(dir, 'deploy.json')
        pipeline_json = os.path.join(dir, 'pipeline.json')
        detail_json = os.path.join(dir, 'detail.json')
        assert os.path.exists(pipeline_json)
        assert os.path.exists(detail_json)
        assert os.path.exists(deploy_json)


def wrap_target():
    return 0


def test_target_wrapper():

    log_level = logging.INFO

    ret_value = mp.Value('d', 0, lock=False)
    ret_value.value = -1
    wrap_func = partial(target_wrapper, wrap_target, log_level, ret_value)

    process = mp.Process(target=wrap_func)
    process.start()
    process.join()

    assert ret_value.value == 0


def test_get_root_logger():
    from mmdeploy.utils import get_root_logger
    logger = get_root_logger()
    logger.info('This is a test message')


def test_get_library_version():
    assert util.get_library_version('abcdefg') is None
    try:
        lib = importlib.import_module('setuptools')
    except ImportError:
        pass
    else:
        assert util.get_library_version('setuptools') == lib.__version__


def test_get_codebase_version():
    versions = util.get_codebase_version()
    for k, v in versions.items():
        assert v == util.get_library_version(k)


def test_get_backend_version():
    versions = util.get_backend_version()
    for k, v in versions.items():
        assert v == util.get_library_version(k)
