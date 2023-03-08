# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.tensorrt import TensorRTManager as backend_mgr
from mmdeploy.backend.tensorrt import TensorRTParam

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)

_extension = '.engine'


class TestBackendParam:

    def test_get_model_files(self):
        param = TensorRTParam(work_dir='', file_name='tmp')
        assert param.file_name == 'tmp' + _extension

        assert param.get_model_files() == 'tmp' + _extension

    def test_check_param(self):
        with pytest.raises(ValueError):
            param = TensorRTParam(int8_mode=True, int8_algorithm='invalid')
            param.check_param()

    def test_parse_args(self):
        parser = argparse.ArgumentParser()
        TensorRTParam.add_arguments(parser)
        args = parser.parse_args([
            '--fp16-mode', '--int8-mode', '--int8-algorithm', 'maxmin',
            '--max-workspace-size', '1024'
        ])

        assert getattr(args, 'device', None) == 'cuda'
        assert getattr(args, 'fp16_mode', None) is True
        assert getattr(args, 'int8_mode', None) is True
        assert getattr(args, 'int8_algorithm', None) == 'maxmin'
        assert getattr(args, 'max_workspace_size', None) == 1024


class TestManager:

    @pytest.fixture(scope='class')
    def inputs(self, input_dict_2i):
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('torch cuda is not available')
        cuda_inputs = dict((k, v.cuda()) for k, v in input_dict_2i.items())
        yield cuda_inputs

    @pytest.fixture(scope='class')
    def outputs(self, output_dict_2i2o):
        torch = pytest.importorskip('torch')
        if not torch.cuda.is_available():
            pytest.skip('torch cuda is not available')
        cuda_outputs = dict((k, v.cuda()) for k, v in output_dict_2i2o.items())
        yield cuda_outputs

    @pytest.fixture(scope='class')
    def input_shape_dict(self, input_shape_dict_2i):
        yield input_shape_dict_2i

    @pytest.fixture(scope='class')
    def onnx_model(self, onnx_model_dynamic_2i2o):
        yield onnx_model_dynamic_2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, input_shape_dict, onnx_model_dynamic_2i2o):
        from tempfile import NamedTemporaryFile
        save_path = NamedTemporaryFile(suffix='.engine').name
        backend_mgr.to_backend(onnx_model_dynamic_2i2o, save_path,
                               input_shape_dict)
        yield save_path

    def test_to_backend(self, backend_model):
        assert osp.exists(backend_model)

    def test_to_backend_from_param(self, input_shape_dict, onnx_model):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(
                work_dir=work_dir,
                file_name='tmp',
                input_shapes=input_shape_dict)
            backend_mgr.to_backend_from_param(onnx_model, param)
            assert osp.exists(param.get_model_files())

    def test_to_backend_from_param_quanti(self, input_shape_dict, onnx_model,
                                          inputs):

        def _quanti_data():
            yield inputs

        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(
                work_dir=work_dir,
                file_name='tmp',
                input_shapes=input_shape_dict,
                int8_mode=True,
                quanti_data=_quanti_data())
            backend_mgr.to_backend_from_param(onnx_model, param)
            assert osp.exists(param.get_model_files())

    def test_build_wrapper(self, backend_model, inputs, outputs,
                           assert_forward):
        wrapper = backend_mgr.build_wrapper(backend_model)
        assert_forward(wrapper, inputs, outputs)

    def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
                                      assert_forward):
        param = backend_mgr.build_param(work_dir='', file_name=backend_model)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        assert_forward(wrapper, inputs, outputs)

    def test_parse_args(self, onnx_model, input_shape_dict):
        # make input shapes
        input_shapes = []
        for name, shape in input_shape_dict.items():
            shape = 'x'.join(str(i) for i in shape)
            input_shapes.append(f'{name}:{shape}')
        input_shapes = ','.join(input_shapes)

        with TemporaryDirectory() as work_dir:

            # make args
            args = ['convert']
            args += ['--onnx-path', onnx_model]
            args += ['--work-dir', work_dir]
            args += ['--file-name', 'tmp']
            args += ['--input-shapes', input_shapes]

            parser = argparse.ArgumentParser()
            generator = backend_mgr.parse_args(parser, args=args)

            try:
                next(generator)
                next(generator)
            except StopIteration:
                assert osp.exists(osp.join(work_dir, 'tmp' + _extension))
