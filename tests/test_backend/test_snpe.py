# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.snpe import SNPEManager as backend_mgr
from mmdeploy.backend.snpe import SNPEParam

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)

_extension = '.dlc'


class TestBackendParam:

    def test_get_model_files(self):
        param = SNPEParam(work_dir='', file_name='tmp')
        assert param.file_name == 'tmp' + _extension

        assert param.get_model_files() == 'tmp' + _extension


class TestManager:

    @pytest.fixture(scope='class')
    def inputs(self, input_dict_2i):
        yield input_dict_2i

    @pytest.fixture(scope='class')
    def outputs(self, output_dict_2i2o):
        yield output_dict_2i2o

    @pytest.fixture(scope='class')
    def ir_model(self, onnx_model_static_2i2o):
        yield onnx_model_static_2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, ir_model):
        with TemporaryDirectory() as tmp_dir:
            model_path = osp.join(tmp_dir, 'tmp' + _extension)
            backend_mgr.to_backend(ir_model, model_path)

            yield model_path

    def test_to_backend(self, backend_model):
        assert osp.exists(backend_model)

    def test_to_backend_from_param(self, ir_model):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(work_dir=work_dir, file_name='tmp')
            backend_mgr.to_backend_from_param(ir_model, param)

            param_path = param.get_model_files()
            assert osp.exists(param_path)

    def test_parse_args(self, ir_model):

        with TemporaryDirectory() as work_dir:
            param_name = 'tmp' + _extension
            # make args
            args = ['convert']
            args += ['--onnx-path', ir_model]
            args += ['--work-dir', work_dir]
            args += ['--file-name', param_name]

            parser = argparse.ArgumentParser()
            with backend_mgr.parse_args(parser, args=args):
                pass
            assert osp.exists(osp.join(work_dir, param_name))
