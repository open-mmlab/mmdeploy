# Copyright (c) OpenMMLab. All rights reserved.
import os
import subprocess
import tempfile

import mmcv
import onnx
import torch

import mmdeploy.apis.tensorrt as trt_apis
from mmdeploy.utils import Backend
from mmdeploy.utils.test import assert_allclose, check_backend


class TestOnnxRTExporter:
    __test__ = False

    def __init__(self):
        self.backend_name = 'onnxruntime'

    def check_env(self):
        check_backend(Backend.ONNXRUNTIME, True)

    def run_and_validate(self,
                         model,
                         input_list,
                         model_name='tmp',
                         tolerate_small_mismatch=False,
                         do_constant_folding=True,
                         dynamic_axes=None,
                         output_names=None,
                         input_names=None,
                         expected_result=None,
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
        if expected_result is None:
            with torch.no_grad():
                model_outputs = model(*input_list)
        else:
            model_outputs = expected_result
        if isinstance(model_outputs, torch.Tensor):
            model_outputs = [model_outputs]
        else:
            model_outputs = list(model_outputs)

        from mmdeploy.backend.onnxruntime import ORTWrapper
        onnx_model = ORTWrapper(onnx_file_path, 'cpu', output_names)
        with torch.no_grad():
            onnx_outputs = onnx_model.forward(
                dict(zip(input_names, input_list)))
        onnx_outputs = [onnx_outputs[i] for i in output_names]
        assert_allclose(model_outputs, onnx_outputs, tolerate_small_mismatch)


class TestTensorRTExporter:
    __test__ = False

    def __init__(self):
        self.backend_name = 'tensorrt'

    def check_env(self):
        check_backend(Backend.TENSORRT, True)

    def run_and_validate(self,
                         model,
                         input_list,
                         model_name='tmp',
                         tolerate_small_mismatch=False,
                         do_constant_folding=True,
                         dynamic_axes=None,
                         output_names=None,
                         input_names=None,
                         expected_result=None,
                         save_dir=None):
        if save_dir is None:
            onnx_file_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
            trt_file_path = tempfile.NamedTemporaryFile(suffix='.engine').name
        else:
            os.makedirs(save_dir, exist_ok=True)
            onnx_file_path = os.path.join(save_dir, model_name + '.onnx')
            trt_file_path = os.path.join(save_dir, model_name + '.engine')
        input_list = [data.cuda() for data in input_list]
        if isinstance(model, onnx.onnx_ml_pb2.ModelProto):
            onnx.save(model, onnx_file_path)
        else:
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
                        fp16_mode=False, max_workspace_size=1 << 28),
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
        work_dir, filename = os.path.split(trt_file_path)
        trt_apis.onnx2tensorrt(
            work_dir,
            filename,
            0,
            deploy_cfg=deploy_cfg,
            onnx_model=onnx_model)
        if expected_result is None and not isinstance(
                model, onnx.onnx_ml_pb2.ModelProto):
            with torch.no_grad():
                model_outputs = model(*input_list)
        else:
            model_outputs = expected_result
        if isinstance(model_outputs, torch.Tensor):
            model_outputs = [model_outputs.cpu().float()]
        else:
            model_outputs = [data.cpu().float() for data in model_outputs]

        from mmdeploy.backend.tensorrt import TRTWrapper
        trt_model = TRTWrapper(trt_file_path, output_names)
        trt_outputs = trt_model(dict(zip(input_names, input_list)))
        trt_outputs = [trt_outputs[i].float().cpu() for i in output_names]
        assert_allclose(model_outputs, trt_outputs, tolerate_small_mismatch)


class TestNCNNExporter:
    __test__ = False

    def __init__(self):
        self.backend_name = 'ncnn'

    def check_env(self):
        check_backend(Backend.NCNN, True)

    def run_and_validate(self,
                         model,
                         inputs_list,
                         model_name='tmp',
                         tolerate_small_mismatch=False,
                         do_constant_folding=True,
                         dynamic_axes=None,
                         output_names=None,
                         input_names=None,
                         save_dir=None):
        if save_dir is None:
            onnx_file_path = tempfile.NamedTemporaryFile().name
            ncnn_param_path = tempfile.NamedTemporaryFile().name
            ncnn_bin_path = tempfile.NamedTemporaryFile().name
        else:
            onnx_file_path = os.path.join(save_dir, model_name + '.onnx')
            ncnn_param_path = os.path.join(save_dir, model_name + '.param')
            ncnn_bin_path = os.path.join(save_dir, model_name + '.bin')

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

        from mmdeploy.backend.ncnn.init_plugins import get_onnx2ncnn_path
        onnx2ncnn_path = get_onnx2ncnn_path()
        subprocess.call(
            [onnx2ncnn_path, onnx_file_path, ncnn_param_path, ncnn_bin_path])

        with torch.no_grad():
            model_outputs = model(*inputs_list)
        if isinstance(model_outputs, torch.Tensor):
            model_outputs = [model_outputs]
        else:
            model_outputs = list(model_outputs)
        model_outputs = [
            model_output.float() for model_output in model_outputs
        ]

        from mmdeploy.backend.ncnn import NCNNWrapper
        ncnn_model = NCNNWrapper(ncnn_param_path, ncnn_bin_path, output_names)
        ncnn_outputs = ncnn_model(dict(zip(input_names, inputs_list)))
        ncnn_outputs = [ncnn_outputs[name] for name in output_names]

        if model_name.startswith('topk_no_sorted'):
            dim = int(model_name.split('_')[-1])
            model_outputs = torch.stack(model_outputs, dim=-1).\
                sort(dim=dim).values
            ncnn_outputs = torch.stack(ncnn_outputs, dim=-1).\
                sort(dim=dim).values
            assert_allclose([model_outputs], [ncnn_outputs],
                            tolerate_small_mismatch)
        else:
            assert_allclose(model_outputs, ncnn_outputs,
                            tolerate_small_mismatch)

    def onnx2ncnn(self, model, model_name, output_names, save_dir=None):

        def _from_onnx(self, model, model_name, output_names, save_dir=None):
            onnx_file_path = os.path.join(save_dir, model_name + '.onnx')
            ncnn_param_path = os.path.join(save_dir, model_name + '.param')
            ncnn_bin_path = os.path.join(save_dir, model_name + '.bin')

            onnx.save_model(model, onnx_file_path)

            from mmdeploy.backend.ncnn import from_onnx
            from_onnx(onnx_file_path, os.path.join(save_dir, model_name))

            from mmdeploy.backend.ncnn import NCNNWrapper
            ncnn_model = NCNNWrapper(ncnn_param_path, ncnn_bin_path,
                                     output_names)

            return ncnn_model

        if save_dir is None:
            with tempfile.TemporaryDirectory() as save_dir:
                return _from_onnx(
                    self, model, model_name, output_names, save_dir=save_dir)
        else:
            return _from_onnx(
                self, model, model_name, output_names, save_dir=save_dir)
