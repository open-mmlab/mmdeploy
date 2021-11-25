import os
import tempfile

import mmcv
import pytest

import mmdeploy.utils as util
from mmdeploy.utils.constants import Backend, Codebase, Task
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
                cfg = mmcv.Config.fromfile(v[1])
            else:
                cfg = v[1]
            assert v[0]._cfg_dict == cfg._cfg_dict


class TestGetTaskType:

    def test_get_task_type_none(self):
        assert util.get_task_type(mmcv.Config(dict())) is None

    def test_get_task_type_default(self):
        assert util.get_task_type(mmcv.Config(dict()),
                                  Task.SUPER_RESOLUTION) == \
            Task.SUPER_RESOLUTION

    def test_get_task_type(self):
        assert util.get_task_type(correct_deploy_path) == Task.SUPER_RESOLUTION


class TestGetCodebase:

    def test_get_codebase_none(self):
        assert util.get_codebase(mmcv.Config(dict())) is None

    def test_get_codebase_default(self):
        assert util.get_codebase(mmcv.Config(dict()),
                                 Codebase.MMEDIT) == Codebase.MMEDIT

    def test_get_codebase(self):
        assert util.get_codebase(correct_deploy_path) == Codebase.MMEDIT


class TestGetBackend:

    def test_get_backend_none(self):
        assert util.get_backend(mmcv.Config(dict())) is None

    def test_get_backend_default(self):
        assert util.get_backend(empty_file_path,
                                Backend.ONNXRUNTIME) == Backend.ONNXRUNTIME

    def test_get_backend(self):
        assert util.get_backend(correct_deploy_path) == Backend.ONNXRUNTIME


class TestGetOnnxConfig:

    def test_get_onnx_config_error(self):
        with pytest.raises(Exception):
            util.get_onnx_config(empty_file_path)

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

    config_with_onnx_config = mmcv.Config(dict(onnx_config=dict()))

    config_with_dynamic_axes = mmcv.Config(
        dict(
            onnx_config=dict(
                dynamic_axes={'input': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'
                }})))

    def test_is_dynamic_batch_none(self):
        assert util.is_dynamic_batch(
            TestIsDynamic.config_with_onnx_config) is False

    def test_is_dynamic_batch_error_name(self):
        assert util.is_dynamic_batch(TestIsDynamic.config_with_dynamic_axes,
                                     'output') is False

    def test_is_dynamic_batch(self):
        assert util.is_dynamic_batch(
            TestIsDynamic.config_with_dynamic_axes) is True

    def test_is_dynamic_shape_none(self):
        assert util.is_dynamic_shape(
            TestIsDynamic.config_with_onnx_config) is False

    def test_is_dynamic_shape_error_name(self):
        assert util.is_dynamic_shape(TestIsDynamic.config_with_dynamic_axes,
                                     'output') is False

    def test_is_dynamic_shape(self):
        assert util.is_dynamic_shape(
            TestIsDynamic.config_with_dynamic_axes) is True


class TestGetInputShape:
    config_without_input_shape = mmcv.Config(
        dict(onnx_config=dict(input_shape=None)))
    config_with_input_shape = mmcv.Config(
        dict(onnx_config=dict(input_shape=[1, 1])))
    config_with_error_shape = mmcv.Config(
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

    config_with_mask = mmcv.Config(
        dict(partition_config=dict(apply_marks=True)))

    def test_cfg_apply_marks_none(self):
        assert util.cfg_apply_marks(mmcv.Config(dict())) is None

    def test_cfg_apply_marks(self):
        assert util.cfg_apply_marks(TestCfgApplyMark.config_with_mask) is True


class TestGetPartitionConfig:

    config_with_mask = mmcv.Config(
        dict(partition_config=dict(apply_marks=True)))
    config_without_mask = mmcv.Config(
        dict(partition_config=dict(apply_marks=False)))

    def test_get_partition_config_none(self):
        assert util.get_partition_config(mmcv.Config(dict())) is None

    def test_get_partition_config_without_mask(self):
        assert util.get_partition_config(
            TestGetPartitionConfig.config_without_mask) is None

    def test_get_partition_config(self):
        assert util.get_partition_config(
            TestGetPartitionConfig.config_with_mask) == dict(apply_marks=True)


class TestGetCalib:
    config_with_calib = mmcv.Config(
        dict(calib_config=dict(create_calib=True, calib_file='calib_data.h5')))

    config_without_calib = mmcv.Config(
        dict(
            calib_config=dict(create_calib=False, calib_file='calib_data.h5')))

    def test_get_calib_config(self):
        assert util.get_calib_config(TestGetCalib.config_with_calib) == dict(
            create_calib=True, calib_file='calib_data.h5')

    def test_get_calib_filename_none(self):
        assert util.get_calib_filename(mmcv.Config(dict())) is None

    def test_get_calib_filename_false(self):
        assert util.get_calib_filename(
            TestGetCalib.config_without_calib) is None

    def test_get_calib_filename(self):
        assert util.get_calib_filename(
            TestGetCalib.config_with_calib) == 'calib_data.h5'


class TestGetCommonConfig:
    config_with_common_config = mmcv.Config(
        dict(
            backend_config=dict(
                type='tensorrt', common_config=dict(fp16_mode=False))))

    def test_get_common_config(self):
        assert util.get_common_config(
            TestGetCommonConfig.config_with_common_config) == dict(
                fp16_mode=False)


class TestGetModelInputs:

    config_with_model_inputs = mmcv.Config(
        dict(backend_config=dict(model_inputs=[dict(input_shapes=None)])))

    def test_model_inputs(self):
        assert util.get_model_inputs(
            TestGetModelInputs.config_with_model_inputs) == [
                dict(input_shapes=None)
            ]


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
