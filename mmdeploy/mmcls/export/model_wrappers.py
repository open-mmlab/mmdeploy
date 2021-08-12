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
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_file)

        providers = ['CPUExecutionProvider']
        options = [{}]
        is_cuda_available = ort.get_device() == 'GPU'
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            options.insert(0, {'device_id': device_id})
        sess.set_providers(providers, options)

        self.sess = sess
        self.io_binding = sess.io_binding()
        self.output_names = [_.name for _ in sess.get_outputs()]
        self.is_cuda_available = is_cuda_available

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs
        # set io binding for inputs/outputs
        device_type = 'cuda' if self.is_cuda_available else 'cpu'
        if not self.is_cuda_available:
            input_data = input_data.cpu()
        self.io_binding.bind_input(
            name='input',
            device_type=device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=input_data.shape,
            buffer_ptr=input_data.data_ptr())

        for name in self.output_names:
            self.io_binding.bind_output(name)
        # run session to get outputs
        self.sess.run_with_iobinding(self.io_binding)
        results = self.io_binding.copy_outputs_to_cpu()[0]
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
        import ncnn
        from mmdeploy.apis.ncnn import ncnn_ext
        self.net = ncnn.Net()
        ncnn_ext.register_mm_custom_layers(self.net)
        self.net.load_param(ncnn_param_file)
        self.net.load_model(ncnn_bin_file)

    def forward_test(self, imgs, *args, **kwargs):
        import ncnn
        assert len(imgs.shape) == 4
        # Only for batch == 1 now.
        assert imgs.shape[0] == 1
        input_data = imgs[0].cpu().numpy()
        input_data = ncnn.Mat(input_data)
        if self.device_id == -1:
            ex = self.net.create_extractor()
            ex.input('input', input_data)
            ret, results = ex.extract('output')
            results = np.array(results)
            assert ret != -100, 'Memory allocation failed in ncnn layers'
            assert ret == 0
            return [results]
        else:
            raise NotImplementedError('GPU device is not implemented.')


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
