import os
import tempfile

import mmcv
import onnx
import pytest
import torch

import mmdeploy.apis.onnxruntime as ort_apis
import mmdeploy.apis.tensorrt as trt_apis
from mmdeploy.utils.test import assert_allclose


class TestOnnxRTExporter:

    def check_env(self):
        if not ort_apis.is_available():
            pytest.skip('Custom ops of ONNXRuntime are not compiled.')

    def run_and_validate(self,
                         model,
                         inputs_list,
                         model_name='tmp',
                         tolerate_small_mismatch=False,
                         do_constant_folding=True,
                         dynamic_axes=None,
                         output_names=None,
                         input_names=None,
                         work_dir=None):

        if not work_dir:
            onnx_file_path = tempfile.NamedTemporaryFile().name
        else:
            onnx_file_path = os.path.join(work_dir, model_name + '.onnx')

        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(inputs_list),
                onnx_file_path,
                export_params=True,
                keep_initializers_as_inputs=True,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                opset_version=11)

        with torch.no_grad():
            model_outputs = model(*inputs_list)

        if isinstance(model_outputs, torch.Tensor):
            model_outputs = [model_outputs]
        else:
            model_outputs = list(model_outputs)

        onnx_model = ort_apis.ORTWrapper(onnx_file_path, 0, output_names)
        with torch.no_grad():
            onnx_outputs = onnx_model.forward(
                dict(zip(input_names, inputs_list)))
        assert_allclose(model_outputs, onnx_outputs, tolerate_small_mismatch)


class TestTensorRTExporter:

    def check_env(self):
        if not trt_apis.is_available():
            pytest.skip(
                'TensorRT is not installed or custom ops are not compiled.')
        if not torch.cuda.is_available():
            pytest.skip('CUDA is not available.')

    def run_and_validate(self,
                         model,
                         inputs_list,
                         model_name='tmp',
                         tolerate_small_mismatch=False,
                         do_constant_folding=True,
                         dynamic_axes=None,
                         output_names=None,
                         input_names=None,
                         work_dir=None):
        if not work_dir:
            onnx_file_path = tempfile.NamedTemporaryFile().name
            trt_file_path = tempfile.NamedTemporaryFile().name
        else:
            onnx_file_path = os.path.join(work_dir, model_name + '.onnx')
            trt_file_path = os.path.join(work_dir, model_name + '.trt')
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(inputs_list),
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
                                    for data in inputs_list
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
            model_outputs = model(*inputs_list)

        inputs_list = [data.cuda() for data in inputs_list]
        if isinstance(model_outputs, torch.Tensor):
            model_outputs = [model_outputs.cuda()]
        else:
            model_outputs = [tensor.cuda() for tensor in model_outputs]
        trt_model = trt_apis.TRTWrapper(trt_file_path)
        with torch.no_grad():
            trt_outputs = trt_model(dict(zip(input_names, inputs_list)))
        trt_outputs = [trt_outputs[name] for name in output_names]
        assert_allclose(model_outputs, trt_outputs, tolerate_small_mismatch)
