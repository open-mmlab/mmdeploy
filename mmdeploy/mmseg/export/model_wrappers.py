import os.path as osp
import warnings
from typing import Sequence

import numpy as np
import torch
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize


class DeployBaseSegmentor(BaseSegmentor):

    def __init__(self, class_names: Sequence[str], device_id: int):
        super(DeployBaseSegmentor, self).__init__(init_cfg=None)
        self.CLASSES = class_names
        self.device_id = device_id
        self.PALETTE = None

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def encode_decode(self, img, img_metas):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def simple_test(self, img, img_meta, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward(self, img, img_metas, **kwargs):
        seg_pred = self.forward_test(img, img_metas, **kwargs)
        # whole mode supports dynamic shape
        ori_shape = img_metas[0][0]['ori_shape']
        if not (ori_shape[0] == seg_pred.shape[-2]
                and ori_shape[1] == seg_pred.shape[-1]):
            seg_pred = torch.from_numpy(seg_pred).float()
            seg_pred = resize(
                seg_pred, size=tuple(ori_shape[:2]), mode='nearest')
            seg_pred = seg_pred.long().detach().cpu().numpy()
        # remove unnecessary dim
        seg_pred = seg_pred.squeeze(1)
        seg_pred = list(seg_pred)
        return seg_pred


class ONNXRuntimeSegmentor(DeployBaseSegmentor):

    def __init__(self, onnx_file: str, class_names: Sequence[str],
                 device_id: int):
        super(ONNXRuntimeSegmentor, self).__init__(class_names, device_id)

        import onnxruntime as ort
        from mmdeploy.apis.onnxruntime import get_ops_path

        # get the custom op path
        ort_custom_op_path = get_ops_path()
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        sess = ort.InferenceSession(onnx_file, session_options)
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
        for name in self.output_names:
            self.io_binding.bind_output(name)

    def forward_test(self, imgs, img_metas, **kwargs):
        input_data = imgs[0]
        device_type = input_data.device.type
        self.io_binding.bind_input(
            name='input',
            device_type=device_type,
            device_id=self.device_id,
            element_type=np.float32,
            shape=input_data.shape,
            buffer_ptr=input_data.data_ptr())
        self.sess.run_with_iobinding(self.io_binding)
        seg_pred = self.io_binding.copy_outputs_to_cpu()[0]
        return seg_pred


class TensorRTSegmentor(DeployBaseSegmentor):

    def __init__(self, trt_file: str, class_names: Sequence[str],
                 device_id: int):
        super(TensorRTSegmentor, self).__init__(class_names, device_id)

        from mmdeploy.apis.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom plugins, \
                you may have to build backend ops with TensorRT')
        model = TRTWrapper(trt_file)
        self.model = model
        self.output_name = self.model.output_names[0]

    def forward_test(self, imgs, img_metas, **kwargs):
        input_data = imgs[0].contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            seg_pred = self.model({'input': input_data})[self.output_name]
        seg_pred = seg_pred.detach().cpu().numpy()
        return seg_pred
