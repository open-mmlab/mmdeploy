# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import pytest

from mmdeploy.backend.tensorrt import TensorRTBackendParam, TensorRTManager

try:
    from torch.testing import assert_close as torch_assert_close
except Exception:
    from torch.testing import assert_allclose as torch_assert_close


class TestTensorrtBackendParam:

    def test_check_param(self):
        with pytest.raises(ValueError):
            param = TensorRTBackendParam(
                int8_mode=True, int8_algorithm='invalid')
            param.check_param()

    def test_parse_args(self):
        parser = argparse.ArgumentParser()
        TensorRTBackendParam.add_arguments(parser)
        args = parser.parse_args([
            '--fp16-mode', '--int8-mode', '--int8-algorithm', 'maxmin',
            '--max-workspace-size', '1024'
        ])

        assert getattr(args, 'device', None) == 'cuda'
        assert getattr(args, 'fp16_mode', None) is True
        assert getattr(args, 'int8_mode', None) is True
        assert getattr(args, 'int8_algorithm', None) == 'maxmin'
        assert getattr(args, 'max_workspace_size', None) == 1024


class TestTensorRTManager:

    def _test_forward(self, wrapper, inputs, gts):
        outputs = wrapper(inputs)
        for name in outputs:
            out = outputs[name]
            gt = gts[name]
            torch_assert_close(out, gt)

    _trt = pytest.importorskip('tensorrt')

    @pytest.fixture(scope='class')
    def backend_mgr(self):
        yield TensorRTManager

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
    def param(self, backend_mgr):
        backend_mgr.build_param(work_dir='')

    @pytest.fixture(scope='class')
    def input_shape_dict(self, input_shape, input_names_2i):
        yield dict(zip(input_names_2i, [input_shape] * 2))

    @pytest.fixture(scope='class')
    def onnx_model(self, onnx_model_dynamic_2i2o):
        yield onnx_model_dynamic_2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, backend_mgr, input_shape_dict,
                      onnx_model_dynamic_2i2o):
        from tempfile import NamedTemporaryFile
        save_path = NamedTemporaryFile(suffix='.engine').name
        backend_mgr.to_backend(onnx_model_dynamic_2i2o, save_path,
                               input_shape_dict)
        yield save_path

    def test_to_backend(self, backend_model):
        assert osp.exists(backend_model)

    def test_to_backend_from_param(self, backend_mgr, tmp_path,
                                   input_shape_dict, onnx_model):
        save_path = str(tmp_path / 'tmp.engine')
        param = backend_mgr.build_param(
            work_dir='', file_name=save_path, input_shapes=input_shape_dict)
        backend_mgr.to_backend_from_param(onnx_model, param)
        assert osp.exists(save_path)

    def test_build_wrapper(self, backend_mgr, backend_model, inputs, outputs):
        wrapper = backend_mgr.build_wrapper(backend_model)
        self._test_forward(wrapper, inputs, outputs)

    def test_build_wrapper_from_param(self, backend_mgr, backend_model, inputs,
                                      outputs):
        param = backend_mgr.build_param(work_dir='', file_name=backend_model)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        self._test_forward(wrapper, inputs, outputs)

    def test_parse_args(self, backend_mgr, onnx_model, tmp_path,
                        input_shape_dict):
        # make input shapes
        input_shapes = []
        for name, shape in input_shape_dict.items():
            shape = 'x'.join(str(i) for i in shape)
            input_shapes.append(f'{name}:{shape}')
        input_shapes = ','.join(input_shapes)

        save_path = str(tmp_path / 'tmp.engine')

        # make args
        args = ['convert']
        args += ['--onnx-path', onnx_model]
        args += ['--work-dir', '/']
        args += ['--file-name', save_path]
        args += ['--input-shapes', input_shapes]

        parser = argparse.ArgumentParser()
        generator = backend_mgr.parse_args(parser, args=args)

        try:
            next(generator)
            next(generator)
        except StopIteration:
            assert osp.exists(save_path)
