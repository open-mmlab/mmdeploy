# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.openvino import OpenVINOManager as backend_mgr
from mmdeploy.backend.openvino import OpenVINOParam

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)

_extension = '.xml'
_bin_extension = '.bin'


class TestBackendParam:

    def test_get_model_files(self):
        param = OpenVINOParam(work_dir='', file_name='tmp')
        assert param.file_name == 'tmp' + _extension
        assert param.bin_name == 'tmp' + _bin_extension

        assert param.get_model_files() == ('tmp' + _extension,
                                           'tmp' + _bin_extension)


class TestManager:

    @pytest.fixture(scope='class')
    def inputs(self, input_dict_2i):
        yield input_dict_2i

    @pytest.fixture(scope='class')
    def outputs(self, output_dict_2i2o):
        yield output_dict_2i2o

    @pytest.fixture(scope='class')
    def input_shape_dict(self, input_shape_dict_2i):
        yield input_shape_dict_2i

    @pytest.fixture(scope='class')
    def output_names(self, output_names_2i2o):
        yield output_names_2i2o

    @pytest.fixture(scope='class')
    def onnx_model(self, onnx_model_dynamic_2i2o):
        yield onnx_model_dynamic_2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, onnx_model, input_shape_dict, output_names):
        with TemporaryDirectory() as tmp_dir:
            param_path = osp.join(tmp_dir, 'tmp' + _extension)
            bin_path = osp.join(tmp_dir, 'tmp' + _bin_extension)
            backend_mgr.to_backend(
                onnx_model,
                param_path,
                input_info=input_shape_dict,
                output_names=output_names)

            yield param_path, bin_path

    def test_to_backend(self, backend_model):
        for path in backend_model:
            assert osp.exists(path)

    def test_to_backend_from_param(self, onnx_model):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(work_dir=work_dir, file_name='tmp')
            backend_mgr.to_backend_from_param(onnx_model, param)

            param_path, bin_path = param.get_model_files()
            assert osp.exists(param_path)
            assert osp.exists(bin_path)

    def test_build_wrapper(self, backend_model, inputs, outputs,
                           assert_forward):
        wrapper = backend_mgr.build_wrapper(*backend_model)
        assert_forward(wrapper, inputs, outputs)

    def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
                                      assert_forward):
        param_path, bin_path = backend_model
        param = backend_mgr.build_param(
            work_dir='', file_name=param_path, bin_name=bin_path)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        assert_forward(wrapper, inputs, outputs)

    def test_parse_args(self, onnx_model, input_shape_dict, output_names):
        # make input shapes
        input_shapes = []
        for name, shape in input_shape_dict.items():
            shape = 'x'.join(str(i) for i in shape)
            input_shapes.append(f'{name}:{shape}')
        input_shapes = ','.join(input_shapes)

        with TemporaryDirectory() as work_dir:
            param_name = 'tmp' + _extension
            # make args
            args = ['convert']
            args += ['--onnx-path', onnx_model]
            args += ['--work-dir', work_dir]
            args += ['--file-name', param_name]
            args += ['--output-names', *output_names]
            args += ['--input-shapes', input_shapes]

            parser = argparse.ArgumentParser()
            generator = backend_mgr.parse_args(parser, args=args)

            try:
                next(generator)
                next(generator)
            except StopIteration:
                pass
            assert osp.exists(osp.join(work_dir, param_name))
            assert osp.exists(osp.join(work_dir, 'tmp' + _bin_extension))
