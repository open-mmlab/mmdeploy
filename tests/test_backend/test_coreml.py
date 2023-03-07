# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.coreml import CoreMLManager as backend_mgr
from mmdeploy.backend.coreml import CoreMLParam

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)


class TestBackendParam:

    def test_get_model_files(self):
        param = CoreMLParam(work_dir='', file_name='tmp')
        assert param.file_name == 'tmp.mlpackage'

        assert param.get_model_files() == 'tmp.mlpackage'


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
    def input_names(self, input_names_2i):
        yield input_names_2i

    @pytest.fixture(scope='class')
    def output_names(self, output_names_2i2o):
        yield output_names_2i2o

    @pytest.fixture(scope='class')
    def ir_model(self, torchscript_model2i2o):
        yield torchscript_model2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, ir_model, input_names, output_names,
                      input_shape_dict):
        with TemporaryDirectory() as tmp_dir:
            model_path = osp.join(tmp_dir, 'tmp.mlpackage')
            backend_mgr.to_backend(
                ir_model,
                model_path,
                input_names=input_names,
                output_names=output_names,
                input_shapes=input_shape_dict)

            yield model_path

    def test_to_backend(self, backend_model):
        assert osp.exists(backend_model)

    def test_to_backend_from_param(self, ir_model, input_names, output_names,
                                   input_shape_dict):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(
                work_dir=work_dir,
                file_name='tmp',
                input_shapes=input_shape_dict,
                input_names=input_names,
                output_names=output_names)
            backend_mgr.to_backend_from_param(ir_model, param)

            param_path = param.get_model_files()
            assert osp.exists(param_path)

    def test_build_wrapper(self, backend_model, inputs, outputs,
                           assert_forward):
        wrapper = backend_mgr.build_wrapper(backend_model)
        assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
                                      assert_forward):
        param = backend_mgr.build_param(work_dir='', file_name=backend_model)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    def test_parse_args(self, ir_model, input_names, output_names,
                        input_shape_dict):

        # make input shapes
        input_shapes = []
        for name, shape in input_shape_dict.items():
            shape = 'x'.join(str(i) for i in shape)
            input_shapes.append(f'{name}:{shape}')
        input_shapes = ','.join(input_shapes)

        with TemporaryDirectory() as work_dir:
            param_name = 'tmp.mlpackage'
            # make args
            args = ['convert']
            args += ['--torchscript-path', ir_model]
            args += ['--work-dir', work_dir]
            args += ['--file-name', param_name]
            args += ['--input-shapes', input_shapes]
            args += ['--input-names', *input_names]
            args += ['--output-names', *output_names]

            parser = argparse.ArgumentParser()
            generator = backend_mgr.parse_args(parser, args=args)

            try:
                next(generator)
                next(generator)
            except StopIteration:
                pass
            assert osp.exists(osp.join(work_dir, param_name))
