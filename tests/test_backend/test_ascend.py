# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.ascend import AscendManager as backend_mgr
from mmdeploy.backend.ascend import AscendParam

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)

_extension = '.om'


class TestBackendParam:

    def test_get_model_files(self):
        param = AscendParam(work_dir='', file_name='tmp')
        assert param.file_name == 'tmp' + _extension

        assert param.get_model_files() == 'tmp' + _extension

    def test_add_argument(self):
        parser = argparse.ArgumentParser()
        AscendParam.add_argument(
            parser, 'dynamic_dims', dtype=str, default=None, desc='')

        assert parser.parse_args(['--dynamic-dims',
                                  '3,2,-1']).dynamic_dims == {
                                      None: (3, 2, -1)
                                  }
        assert parser.parse_args(['--dynamic-dims',
                                  'input:3,2,-1']).dynamic_dims == {
                                      'input': (3, 2, -1)
                                  }
        assert parser.parse_args(
            ['--dynamic-dims', 'input1:3,2,-1',
             'input2:3,2,-1']).dynamic_dims == {
                 'input1': (3, 2, -1),
                 'input2': (3, 2, -1)
             }
        assert parser.parse_args(
            ['--dynamic-dims',
             'input1:3,2,-1;input2:3,2,-1']).dynamic_dims == {
                 'input1': (3, 2, -1),
                 'input2': (3, 2, -1)
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
    def onnx_model(self, onnx_model_static_2i2o):
        yield onnx_model_static_2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, onnx_model, input_shape_dict):
        from mmdeploy.backend.ascend import AtcParam
        with TemporaryDirectory() as tmp_dir:
            model_path = osp.join(tmp_dir, 'tmp' + _extension)
            backend_mgr.to_backend(onnx_model, model_path,
                                   AtcParam(input_shapes=input_shape_dict))

            yield model_path

    def test_to_backend(self, backend_model):
        assert osp.exists(backend_model)

    def test_to_backend_from_param(self, onnx_model, input_shape_dict):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(
                work_dir=work_dir,
                file_name='tmp',
                input_shapes=input_shape_dict)
            backend_mgr.to_backend_from_param(onnx_model, param)

            param_path = param.get_model_files()
            assert osp.exists(param_path)

    def test_build_wrapper(self, backend_model, inputs, outputs,
                           assert_forward):
        wrapper = backend_mgr.build_wrapper(backend_model)
        assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
                                      assert_forward):
        param = backend_mgr.build_param(
            work_dir='', device='cpu', file_name=backend_model)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    def test_parse_args(self, onnx_model, input_shape_dict):

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
            args += ['--input-shapes', input_shapes]

            parser = argparse.ArgumentParser()
            with backend_mgr.parse_args(parser, args=args):
                pass
            assert osp.exists(osp.join(work_dir, param_name))
