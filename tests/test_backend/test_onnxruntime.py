# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

from mmdeploy.backend.onnxruntime import ONNXRuntimeManager

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close


class TestONNXRuntimeManager:

    def _test_forward(self, wrapper, inputs, gts):
        outputs = wrapper(inputs)
        for name in outputs:
            out = outputs[name]
            gt = gts[name]
            torch_assert_close(out, gt)

    _ort = pytest.importorskip('onnxruntime')

    @pytest.fixture(scope='class')
    def backend_mgr(self):
        yield ONNXRuntimeManager

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

    def test_to_backend_from_param(self, backend_mgr, tmp_path, backend_model):
        save_path = str(tmp_path / 'tmp.onnx')
        param = backend_mgr.build_param(work_dir='', file_name=save_path)
        backend_mgr.to_backend_from_param(backend_model, param)
        assert osp.exists(save_path)

    def test_build_wrapper(self, backend_mgr, backend_model, inputs, outputs):
        wrapper = backend_mgr.build_wrapper(backend_model, 'cpu')
        self._test_forward(wrapper, inputs, outputs)

    def test_build_wrapper_from_param(self, backend_mgr, backend_model, inputs,
                                      outputs):
        param = backend_mgr.build_param(work_dir='', file_name=backend_model)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        self._test_forward(wrapper, inputs, outputs)
