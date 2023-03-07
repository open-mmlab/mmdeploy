# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.rknn import RKNNManager as backend_mgr
from mmdeploy.backend.rknn import RKNNParam

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)

device = 'rk1126'


class TestBackendParam:

    def test_get_model_files(self):
        param = RKNNParam(work_dir='', file_name='tmp')
        assert param.file_name == 'tmp.rknn'

        assert param.get_model_files() == 'tmp.rknn'

    def test_add_argument(self):
        parser = argparse.ArgumentParser()
        RKNNParam.add_argument(
            parser, 'mean_values', dtype=str, default=None, desc='')

        assert parser.parse_args(['--mean-values',
                                  '3.2,-2,1']).mean_values == {
                                      None: (3.2, -2.0, 1.0)
                                  }
        assert parser.parse_args(['--mean-values',
                                  'input:3.2,-2,1']).mean_values == {
                                      'input': (3.2, -2.0, 1.0)
                                  }
        assert parser.parse_args(
            ['--mean-values', 'input1:3.2,-2,1',
             'input2:3.2,-2,1']).mean_values == {
                 'input1': (3.2, -2.0, 1.0),
                 'input2': (3.2, -2.0, 1.0)
             }


class TestManager:

    @pytest.fixture(scope='class')
    def inputs(self, input_dict_2i):
        yield input_dict_2i

    @pytest.fixture(scope='class')
    def outputs(self, output_dict_2i2o):
        yield output_dict_2i2o

    @pytest.fixture(scope='class')
    def input_names(self, input_names_2i):
        yield input_names_2i

    @pytest.fixture(scope='class')
    def output_names(self, output_names_2i2o):
        yield output_names_2i2o

    @pytest.fixture(scope='class')
    def input_shape_dict(self, input_shape_dict_2i):
        yield input_shape_dict_2i

    @pytest.fixture(scope='class')
    def onnx_model(self, onnx_model_static_2i2o):
        yield onnx_model_static_2i2o

    @pytest.fixture(scope='class')
    def backend_model(self, onnx_model, input_names, output_names,
                      input_shape_dict):
        from mmdeploy.backend.rknn.onnx2rknn import RKNNConfig
        with TemporaryDirectory() as tmp_dir:
            model_path = osp.join(tmp_dir, 'tmp.rknn')
            backend_mgr.to_backend(
                onnx_model,
                model_path,
                input_names,
                output_names,
                input_shapes=input_shape_dict,
                rknn_config=RKNNConfig(
                    mean_values=[(0, 0, 0), (0, 0, 0)],
                    std_values=[(1, 1, 1), (1, 1, 1)],
                    target_platform=device),
                do_quantization=False)

            yield model_path

    def test_to_backend(self, backend_model):
        assert osp.exists(backend_model[0])

    def test_to_backend_from_param(self, onnx_model, input_names, output_names,
                                   input_shape_dict):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(
                work_dir=work_dir,
                file_name='tmp',
                input_names=input_names,
                output_names=output_names,
                input_shapes=input_shape_dict,
                device=device,
                mean_values=dict(x=(0, 0, 0), y=(0, 0, 0)),
                std_values=dict(x=(1, 1, 1), y=(1, 1, 1)))
            backend_mgr.to_backend_from_param(onnx_model, param)

            param_path = param.get_model_files()
            assert osp.exists(param_path)

    def test_build_wrapper(self, backend_model, inputs, outputs, input_names,
                           output_names, assert_forward):

        wrapper = backend_mgr.build_wrapper(
            backend_model,
            device,
            input_names,
            output_names,
        )
        assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
                                      input_names, output_names,
                                      assert_forward):
        param = backend_mgr.build_param(
            work_dir='',
            device=device,
            file_name=backend_model,
            input_names=input_names,
            output_names=output_names)
        wrapper = backend_mgr.build_wrapper_from_param(param)
        assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    def test_parse_args(self, onnx_model, input_names, output_names,
                        input_shape_dict):

        # make input shapes
        input_shapes = []
        for name, shape in input_shape_dict.items():
            shape = 'x'.join(str(i) for i in shape)
            input_shapes.append(f'{name}:{shape}')
        input_shapes = ','.join(input_shapes)

        with TemporaryDirectory() as work_dir:
            param_name = 'tmp.rknn'
            # make args
            args = ['convert']
            args += ['--onnx-path', onnx_model]
            args += ['--work-dir', work_dir]
            args += ['--file-name', param_name]
            args += ['--input-names', *input_names]
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
