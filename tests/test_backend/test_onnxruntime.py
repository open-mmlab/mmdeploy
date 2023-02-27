# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close


class TestONNXRuntimeManager:
    _ort = pytest.importorskip('onnxruntime')

    @pytest.fixture(scope='class')
    def backend_mgr(self):
        from mmdeploy.backend.onnxruntime import ONNXRuntimeManager
        return ONNXRuntimeManager

    def _test_forward(self, wrapper, inputs, gts):
        outputs = wrapper(inputs)
        for name in outputs:
            out = outputs[name]
            gt = gts[name]
            torch_assert_close(out, gt)

    def test_to_backend_from_param(self, backend_mgr, tmp_path,
                                   onnx_model_dynamic_2i2o):
        tmp_path = str(tmp_path)
        param = backend_mgr.build_param(work_dir='', file_name=tmp_path)
        backend_mgr.to_backend_from_param(onnx_model_dynamic_2i2o, param)
        assert osp.exists(tmp_path)

    def test_build_wrapper(self, backend_mgr, onnx_model_dynamic_2i2o,
                           input_dict_2i, output_dict_2i2o):
        wrapper = backend_mgr.build_wrapper(onnx_model_dynamic_2i2o, 'cpu')
        self._test_forward(wrapper, input_dict_2i, output_dict_2i2o)

    def test_build_wrapper_from_param(self, backend_mgr,
                                      onnx_model_dynamic_2i2o, input_dict_2i,
                                      output_dict_2i2o):
        param = backend_mgr.build_param(
            work_dir='', file_name=onnx_model_dynamic_2i2o)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        self._test_forward(wrapper, input_dict_2i, output_dict_2i2o)
