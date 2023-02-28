# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.ncnn import NCNNBackendParam, NCNNManager

if not NCNNManager.is_available():
    pytest.skip('backend not available')


class TestBackendParam:

    def test_get_model_files(self):
        param = NCNNBackendParam(work_dir='', file_name='tmp')
        assert param.file_name == 'tmp.param'
        assert param.bin_name == 'tmp.bin'

        assert param.get_model_files() == ('tmp.param', 'tmp.bin')


class TestManager:

    @pytest.fixture(scope='class')
    def backend_mgr(self):
        yield NCNNManager

    @pytest.fixture(scope='class')
    def inputs(self, input_dict_2i):
        yield input_dict_2i

    @pytest.fixture(scope='class')
    def outputs(self, output_dict_2i2o):
        yield output_dict_2i2o

    @pytest.fixture(scope='class')
    def onnx_model(self, onnx_model_dynamic_2i2o):
        yield onnx_model_dynamic_2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, backend_mgr, onnx_model):
        with TemporaryDirectory() as tmp_dir:
            param_path = osp.join(tmp_dir, 'tmp.param')
            bin_path = osp.join(tmp_dir, 'tmp.bin')
            backend_mgr.to_backend(onnx_model, param_path, bin_path)

            yield param_path, bin_path

    def test_to_backend(self, backend_model):
        for path in backend_model:
            assert osp.exists(path)

    def test_to_backend_from_param(self, backend_mgr, onnx_model):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(work_dir=work_dir, file_name='tmp')
            backend_mgr.to_backend_from_param(onnx_model, param)

            param_path, bin_path = param.get_model_files()
            assert osp.exists(param_path)
            assert osp.exists(bin_path)

    def test_build_wrapper(self, backend_mgr, backend_model, inputs, outputs,
                           assert_forward):
        wrapper = backend_mgr.build_wrapper(*backend_model)
        assert_forward(wrapper, inputs, outputs)

    def test_build_wrapper_from_param(self, backend_mgr, backend_model, inputs,
                                      outputs, assert_forward):
        param_path, bin_path = backend_model
        param = backend_mgr.build_param(
            work_dir='', file_name=param_path, bin_name=bin_path)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        assert_forward(wrapper, inputs, outputs)

    def test_parse_args(self, backend_mgr, onnx_model):
        with TemporaryDirectory() as work_dir:
            param_name = 'tmp.param'
            # make args
            args = ['convert']
            args += ['--onnx-path', onnx_model]
            args += ['--work-dir', work_dir]
            args += ['--file-name', param_name]

            parser = argparse.ArgumentParser()
            generator = backend_mgr.parse_args(parser, args=args)

            try:
                next(generator)
                next(generator)
            except StopIteration:
                pass
            assert osp.exists(osp.join(work_dir, param_name))
            assert osp.exists(osp.join(work_dir, 'tmp.bin'))
