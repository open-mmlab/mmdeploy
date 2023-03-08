# Copyright (c) OpenMMLab. All rights reserved.
# import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import pytest

from mmdeploy.backend.vacc import VACCManager as backend_mgr
from mmdeploy.backend.vacc import VACCParam

if not backend_mgr.is_available():
    pytest.skip('backend not available', allow_module_level=True)

_extension = '.so'
_json_extension = '.json'
_param_extension = '.params'


class TestBackendParam:

    def test_get_model_files(self):
        param = VACCParam(work_dir='', file_name='tmp')

        assert param.get_model_files() == [
            'tmp-fp16/tmp' + _extension, 'tmp-fp16/tmp' + _json_extension,
            'tmp-fp16/tmp' + _param_extension
        ]


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
    def backend_model(self, onnx_model, input_shape_dict):
        with TemporaryDirectory() as tmp_dir:
            model_name = 'tmp'
            outputs = VACCParam(
                work_dir=tmp_dir,
                file_name=model_name,
                qconfig=dict(dtype='fp16'))
            backend_mgr.to_backend(
                onnx_model,
                tmp_dir,
                model_name,
                input_shapes=input_shape_dict,
                qconfig=dict(dtype='fp16'))

            yield outputs.get_model_files()

    def test_to_backend(self, backend_model):
        for file in backend_model:
            assert osp.exists(file)

    # def test_to_backend_from_param(self, onnx_model, input_names,
    # output_names,
    #                                input_shape_dict):
    #     with TemporaryDirectory() as work_dir:
    #         param = backend_mgr.build_param(
    #             work_dir=work_dir,
    #             file_name='tmp',
    #             input_names=input_names,
    #             output_names=output_names,
    #             input_shapes=input_shape_dict,
    #             device=device,
    #             mean_values=dict(x=(0, 0, 0), y=(0, 0, 0)),
    #             std_values=dict(x=(1, 1, 1), y=(1, 1, 1)))
    #         backend_mgr.to_backend_from_param(onnx_model, param)

    #         param_path = param.get_model_files()
    #         assert osp.exists(param_path)

    # def test_build_wrapper(self, backend_model, inputs, outputs, input_names,
    #                        output_names, assert_forward):

    #     wrapper = backend_mgr.build_wrapper(
    #         backend_model,
    #         device,
    #         input_names,
    #         output_names,
    #     )
    #     assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    # def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
    #                                   input_names, output_names,
    #                                   assert_forward):
    #     param = backend_mgr.build_param(
    #         work_dir='',
    #         device=device,
    #         file_name=backend_model,
    #         input_names=input_names,
    #         output_names=output_names)
    #     wrapper = backend_mgr.build_wrapper_from_param(param)
    #     assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    # def test_parse_args(self, onnx_model, input_names, output_names,
    #                     input_shape_dict):

    #     # make input shapes
    #     input_shapes = []
    #     for name, shape in input_shape_dict.items():
    #         shape = 'x'.join(str(i) for i in shape)
    #         input_shapes.append(f'{name}:{shape}')
    #     input_shapes = ','.join(input_shapes)

    #     with TemporaryDirectory() as work_dir:
    #         param_name = 'tmp' + _extension
    #         # make args
    #         args = ['convert']
    #         args += ['--onnx-path', onnx_model]
    #         args += ['--work-dir', work_dir]
    #         args += ['--file-name', param_name]
    #         args += ['--input-names', *input_names]
    #         args += ['--output-names', *output_names]
    #         args += ['--input-shapes', input_shapes]

    #         parser = argparse.ArgumentParser()
    #         generator = backend_mgr.parse_args(parser, args=args)

    #         try:
    #             next(generator)
    #             next(generator)
    #         except StopIteration:
    #             pass
    #         assert osp.exists(osp.join(work_dir, param_name))
