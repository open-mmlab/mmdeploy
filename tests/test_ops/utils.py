import os
import tempfile

import mmcv
import onnx
import pytest
import torch

import mmdeploy.apis.onnxruntime as ort_apis
import mmdeploy.apis.tensorrt as trt_apis
from mmdeploy.utils.test import assert_allclose

# PytestCollectionWarning: cannot collect test class 'TestOnnxRTExporter'
# because it has a __init__ constructor
pytest.skip('Skip test this file.', allow_module_level=True)


class TestOnnxRTExporter:

    def __init__(self):
        self.backend_name = 'onnxruntime'

    def check_env(self):
        if not ort_apis.is_available():
            pytest.skip('Custom ops of ONNXRuntime are not compiled.')

    def run_and_validate(self,
                         model,
                         input_list,
                         model_name='tmp',
                         tolerate_small_mismatch=False,
                         do_constant_folding=True,
                         dynamic_axes=None,
                         output_names=None,
                         input_names=None,
                         save_dir=None):

        if save_dir is None:
            onnx_file_path = tempfile.NamedTemporaryFile().name
        else:
            onnx_file_path = os.path.join(save_dir, model_name + '.onnx')

        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(input_list),
                onnx_file_path,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                opset_version=11)

        with torch.no_grad():
            model_outputs = model(*input_list)

        if isinstance(model_outputs, torch.Tensor):
            model_outputs = [model_outputs]
        else:
            model_outputs = list(model_outputs)

        onnx_model = ort_apis.ORTWrapper(onnx_file_path, 0, output_names)
        with torch.no_grad():
            onnx_outputs = onnx_model.forward(
                dict(zip(input_names, input_list)))

        assert_allclose(model_outputs, onnx_outputs, tolerate_small_mismatch)


class TestTensorRTExporter:

    def __init__(self):
        self.backend_name = 'tensorrt'

    def check_env(self):
        if not trt_apis.is_available():
            pytest.skip(
                'TensorRT is not installed or custom ops are not compiled.')
        if not torch.cuda.is_available():
            pytest.skip('CUDA is not available.')

    def run_and_validate(self,
                         model,
                         input_list,
                         model_name='tmp',
                         tolerate_small_mismatch=False,
                         do_constant_folding=True,
                         dynamic_axes=None,
                         output_names=None,
                         input_names=None,
                         save_dir=None):
        if save_dir is None:
            onnx_file_path = tempfile.NamedTemporaryFile().name
            trt_file_path = tempfile.NamedTemporaryFile().name
        else:
            onnx_file_path = os.path.join(save_dir, model_name + '.onnx')
            trt_file_path = os.path.join(save_dir, model_name + '.trt')
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(input_list),
                onnx_file_path,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                opset_version=11)

        deploy_cfg = mmcv.Config(
            dict(
                backend_config=dict(
                    type='tensorrt',
                    common_config=dict(
                        fp16_mode=False, max_workspace_size=1 << 30),
                    model_inputs=[
                        dict(
                            input_shapes=dict(
                                zip(input_names, [
                                    dict(
                                        min_shape=data.shape,
                                        opt_shape=data.shape,
                                        max_shape=data.shape)
                                    for data in input_list
                                ])))
                    ])))

        onnx_model = onnx.load(onnx_file_path)
        trt_apis.onnx2tensorrt(
            os.path.dirname(trt_file_path),
            trt_file_path,
            0,
            deploy_cfg=deploy_cfg,
            onnx_model=onnx_model)

        with torch.no_grad():
            model_outputs = model(*input_list)

        if isinstance(model_outputs, torch.Tensor):
            model_outputs = [model_outputs.cpu().float()]
        else:
            model_outputs = [data.cpu().float() for data in model_outputs]

        trt_model = trt_apis.TRTWrapper(trt_file_path)
        inputs_list = [data.cuda() for data in input_list]
        trt_outputs = trt_model(dict(zip(input_names, inputs_list)))
        trt_outputs = [
            trt_outputs[name].cpu().float() for name in output_names
        ]
        assert_allclose(model_outputs, trt_outputs, tolerate_small_mismatch)
