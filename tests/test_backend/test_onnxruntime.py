# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmdeploy.backend.onnxruntime import ONNXRuntimeManager as backend_mgr

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)


class TestManager:

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
    def backend_model(self, onnx_model_dynamic_2i2o):
        yield onnx_model_dynamic_2i2o

    def test_to_backend_from_param(self, tmp_path, backend_model):
        save_path = str(tmp_path / 'tmp.onnx')
        param = backend_mgr.build_param(work_dir='', file_name=save_path)
        backend_mgr.to_backend_from_param(backend_model, param)
        assert osp.exists(save_path)

    def test_build_wrapper(self, backend_model, inputs, outputs,
                           assert_forward):
        wrapper = backend_mgr.build_wrapper(backend_model, 'cpu')
        assert_forward(wrapper, inputs, outputs)

    def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
                                      assert_forward):
        param = backend_mgr.build_param(work_dir='', file_name=backend_model)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        assert_forward(wrapper, inputs, outputs)
