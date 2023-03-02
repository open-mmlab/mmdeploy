# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.tvm import TVMManager as backend_mgr
from mmdeploy.backend.tvm import TVMParam, get_library_ext

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)


class TestBackendParam:

    def test_get_model_files(self):
        param = TVMParam(work_dir='', file_name='tmp')
        assert param.file_name == 'tmp' + get_library_ext()
        assert param.vm_name == 'tmp.vm'

        assert param.get_model_files() == ('tmp' + get_library_ext(), 'tmp.vm')

    def test_add_argument(self):
        parser = argparse.ArgumentParser()
        TVMParam.add_argument(
            parser, 'dtypes', dtype=str, default=None, desc='')

        assert parser.parse_args(['--dtypes', 'float32']).dtypes == {
            None: 'float32'
        }
        assert parser.parse_args(['--dtypes', 'input:float32']).dtypes == {
            'input': 'float32'
        }
        assert parser.parse_args(
            ['--dtypes', 'input1:float32', 'input2:float32']).dtypes == {
                'input1': 'float32',
                'input2': 'float32'
            }
        assert parser.parse_args(['--dtypes',
                                  'input1:float32,input2:float32']).dtypes == {
                                      'input1': 'float32',
                                      'input2': 'float32'
                                  }


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
    def input_dtypes(self):
        yield dict(x='float32', y='float32')

    @pytest.fixture(scope='class')
    def onnx_model(self, onnx_model_dynamic_2i2o):
        yield onnx_model_dynamic_2i2o

    @pytest.fixture(scope='class')
    def output_names(self, output_names_2i2o):
        yield output_names_2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, onnx_model, input_shape_dict, input_dtypes):
        with TemporaryDirectory() as tmp_dir:
            output_path = osp.join(tmp_dir, 'tmp' + get_library_ext())
            backend_mgr.to_backend(
                onnx_model,
                output_path,
                input_shapes=input_shape_dict,
                dtypes=input_dtypes)

            yield output_path

    def test_to_backend(self, backend_model):
        assert osp.exists(backend_model)

    def test_to_backend_from_param(self, onnx_model, input_shape_dict,
                                   input_dtypes):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(
                work_dir=work_dir,
                file_name='tmp',
                input_shapes=input_shape_dict,
                dtypes=input_dtypes)
            backend_mgr.to_backend_from_param(onnx_model, param)

            param_path, _ = param.get_model_files()
            assert osp.exists(param_path)

    def test_build_wrapper(self, backend_model, inputs, outputs, output_names,
                           assert_forward):
        wrapper = backend_mgr.build_wrapper(
            backend_model, output_names=output_names)
        assert_forward(wrapper, inputs, outputs)

    def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
                                      output_names, assert_forward):
        param = backend_mgr.build_param(
            work_dir='', file_name=backend_model, output_names=output_names)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        assert_forward(wrapper, inputs, outputs)

    def test_parse_args(self, onnx_model, input_shape_dict, input_dtypes):
        # make input shapes
        input_shapes = []
        for name, shape in input_shape_dict.items():
            shape = 'x'.join(str(i) for i in shape)
            input_shapes.append(f'{name}:{shape}')
        input_shapes = ','.join(input_shapes)

        dtypes = ','.join(f'{k}:{v}' for k, v in input_dtypes.items())

        with TemporaryDirectory() as work_dir:
            param_name = 'tmp' + get_library_ext()
            # make args
            args = ['convert']
            args += ['--onnx-path', onnx_model]
            args += ['--work-dir', work_dir]
            args += ['--file-name', param_name]
            args += ['--input-shapes', input_shapes]
            args += ['--dtypes', dtypes]

            parser = argparse.ArgumentParser()
            generator = backend_mgr.parse_args(parser, args=args)

            try:
                next(generator)
                next(generator)
            except StopIteration:
                pass
            assert osp.exists(osp.join(work_dir, param_name))
