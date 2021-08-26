import warnings

import numpy as np
import torch
from mmcls.models import BaseClassifier


class DeployBaseClassifier(BaseClassifier):
    """Base Class of Wrapper for classifier's inference."""

    def __init__(self, class_names, device_id):
        super(DeployBaseClassifier, self).__init__()
        self.CLASSES = class_names
        self.device_id = device_id

    def simple_test(self, img, *args, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward_test(self, imgs, *args, **kwargs):
        raise NotImplementedError('This method is not implemented.')


class ONNXRuntimeClassifier(DeployBaseClassifier):
    """Wrapper for classifier's inference with ONNXRuntime."""

    def __init__(self, onnx_file, class_names, device_id):
        super(ONNXRuntimeClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(onnx_file, device_id)

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs
        results = self.model(input_data)[0]
        return list(results)


class TensorRTClassifier(DeployBaseClassifier):

    def __init__(self, trt_file, class_names, device_id):
        super(TensorRTClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom plugins, \
                you may have to build backend ops with TensorRT')
        model = TRTWrapper(trt_file)

        self.model = model

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs
        with torch.cuda.device(self.device_id), torch.no_grad():
            results = self.model({'input': input_data})['output']
        results = results.detach().cpu().numpy()

        return list(results)


class NCNNClassifier(DeployBaseClassifier):

    def __init__(self, ncnn_param_file, ncnn_bin_file, class_names, device_id):
        super(NCNNClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.ncnn import NCNNWrapper
        self.model = NCNNWrapper(
            ncnn_param_file, ncnn_bin_file, output_names=['output'])

    def forward_test(self, imgs, *args, **kwargs):
        results = self.model({'input': imgs})['output']
        results = results.detach().cpu().numpy()
        results_list = list(results)
        return results_list


class PPLClassifier(DeployBaseClassifier):
    """Wrapper for classifier's inference with PPL."""

    def __init__(self, onnx_file, class_names, device_id):
        super(PPLClassifier, self).__init__(class_names, device_id)
        import pyppl.nn as pplnn
        from mmdeploy.apis.ppl import register_engines

        # enable quick select by default to speed up pipeline
        # TODO: open it to users after ppl supports saving serialized models
        # TODO: disable_avx512 will be removed or open to users in config
        engines = register_engines(
            device_id, disable_avx512=False, quick_select=True)
        cuda_options = pplnn.CudaEngineOptions()
        cuda_options.device_id = device_id
        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(
            onnx_file, engines)
        assert runtime_builder is not None, 'Failed to create '\
            'ONNXRuntimeBuilder.'

        runtime_options = pplnn.RuntimeOptions()
        runtime = runtime_builder.CreateRuntime(runtime_options)
        assert runtime is not None, 'Failed to create the instance of Runtime.'

        self.runtime = runtime
        self.CLASSES = class_names
        self.device_id = device_id
        self.inputs = [
            runtime.GetInputTensor(i) for i in range(runtime.GetInputCount())
        ]

    def forward_test(self, imgs, *args, **kwargs):
        import pyppl.common as pplcommon
        input_data = imgs
        self.inputs[0].ConvertFromHost(input_data.cpu().numpy())
        status = self.runtime.Run()
        assert status == pplcommon.RC_SUCCESS, 'Run() '\
            'failed: ' + pplcommon.GetRetCodeStr(status)
        status = self.runtime.Sync()
        assert status == pplcommon.RC_SUCCESS, 'Sync() '\
            'failed: ' + pplcommon.GetRetCodeStr(status)
        results = self.runtime.GetOutputTensor(0).ConvertToHost()
        results = np.array(results, copy=False)

        return list(results)
