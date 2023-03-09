# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from tempfile import TemporaryDirectory

import numpy as np
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


def half_to_uint16(x):
    """Convert a np.float16 number to a uint16 represented."""
    return int(np.frombuffer(np.array(x, dtype=np.float16), dtype=np.uint16))


class TestManager:

    @pytest.fixture(scope='class')
    def inputs(self, input_dict_1i):
        yield input_dict_1i

    @pytest.fixture(scope='class')
    def outputs(self, output_dict_1i1o):
        yield output_dict_1i1o

    @pytest.fixture(scope='class')
    def input_names(self, input_names_1i):
        yield input_names_1i

    @pytest.fixture(scope='class')
    def output_names(self, output_names_1i1o):
        yield output_names_1i1o

    @pytest.fixture(scope='class')
    def input_shape_dict(self, input_shape_dict_1i):
        yield input_shape_dict_1i

    @pytest.fixture(scope='class')
    def onnx_model(self, onnx_model_static_1i1o):
        yield onnx_model_static_1i1o

    @pytest.fixture(scope='class')
    def work_dir(self):
        with TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture(scope='class')
    def backend_model(self, onnx_model, input_shape_dict, work_dir):
        model_name = 'tmp'
        outputs = VACCParam(
            work_dir=work_dir, file_name=model_name, quant_mode='fp16')
        backend_mgr.to_backend(
            onnx_model,
            work_dir,
            model_name,
            input_shapes=input_shape_dict,
            quant_mode='fp16')

        yield outputs.get_model_files()

    @pytest.fixture(scope='class')
    def dummy_vdsp_params_info(self):
        mean = half_to_uint16(0)
        std = half_to_uint16(255)
        return dict(
            vdsp_op_type=300,
            iimage_format=5000,
            iimage_width=8,
            iimage_height=8,
            iimage_width_pitch=8,
            iimage_height_pitch=8,
            short_edge_threshold=8,
            resize_type=1,
            color_cvt_code=2,
            color_space=0,
            crop_size=8,
            meanr=mean,
            meang=mean,
            meanb=mean,
            stdr=std,
            stdg=std,
            stdb=std,
            norma_type=3)

    def test_to_backend(self, backend_model):
        for file in backend_model:
            assert osp.exists(file)

    def test_to_backend_from_param(self, onnx_model, input_shape_dict):
        with TemporaryDirectory() as work_dir:
            param = backend_mgr.build_param(
                work_dir=work_dir,
                file_name='tmp',
                input_shapes=input_shape_dict)
            backend_mgr.to_backend_from_param(onnx_model, param)

            model_files = param.get_model_files()
            for file in model_files:
                assert osp.exists(file)

    # TODO: Enable the test after vdsp parameter available
    # def test_build_wrapper(self, backend_model, inputs, outputs,
    #                        dummy_vdsp_params_info, output_names,
    #                        assert_forward):

    #     wrapper = backend_mgr.build_wrapper(
    #         *backend_model,
    #         dummy_vdsp_params_info,
    #         output_names,
    #     )
    #     assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    # def test_build_wrapper_from_param(self, backend_model, inputs, outputs,
    #                                   dummy_vdsp_params_info, output_names,
    #                                   work_dir, assert_forward):
    #     for file in backend_model:
    #         assert osp.exists(file)

    #     param = backend_mgr.build_param(
    #         work_dir=work_dir,
    #         file_name='tmp',
    #         output_names=output_names,
    #         vdsp_params_info=dummy_vdsp_params_info)
    #     wrapper = backend_mgr.build_wrapper_from_param(param)
    #     assert_forward(wrapper, inputs, outputs, rtol=1e-3, atol=1e-3)

    def test_parse_args(self, onnx_model, output_names, input_shape_dict):

        # make input shapes
        input_shapes = []
        for name, shape in input_shape_dict.items():
            shape = 'x'.join(str(i) for i in shape)
            input_shapes.append(f'{name}:{shape}')
        input_shapes = ','.join(input_shapes)

        with TemporaryDirectory() as work_dir:
            param_name = 'tmp'
            quant_mode = 'fp16'
            # make args
            args = ['convert']
            args += ['--onnx-path', onnx_model]
            args += ['--work-dir', work_dir]
            args += ['--file-name', param_name]
            args += ['--output-names', *output_names]
            args += ['--input-shapes', input_shapes]
            args += ['--quant-mode', quant_mode]

            parser = argparse.ArgumentParser()
            generator = backend_mgr.parse_args(parser, args=args)

            try:
                next(generator)
                next(generator)
            except StopIteration:
                pass
            assert osp.exists(
                osp.join(work_dir, param_name + '-' + quant_mode,
                         param_name + _extension))
            assert osp.exists(
                osp.join(work_dir, param_name + '-' + quant_mode,
                         param_name + _json_extension))
            assert osp.exists(
                osp.join(work_dir, param_name + '-' + quant_mode,
                         param_name + _param_extension))
