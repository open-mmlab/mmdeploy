import warnings

import numpy as np
import torch
from mmedit.core import psnr, ssim, tensor2img
from mmedit.models import BaseModel

from mmdeploy.utils.config_utils import Backend, get_backend, load_config


class DeployBaseRestorer(BaseModel):
    """Base Class of Wrapper for restorer's inference."""

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, device_id, test_cfg=None, **kwargs):
        super(DeployBaseRestorer, self).__init__(**kwargs)
        self.test_cfg = test_cfg
        self.device_id = device_id

    def init_weights(self):
        raise NotImplementedError('This method is not implemented.')

    def forward(self, lq, test_mode=False, **kwargs):
        if (test_mode):
            return self.forward_test(lq, **kwargs)
        else:
            return self.forward_dummy(lq, **kwargs)

    def forward_train(self, imgs, labels):
        raise NotImplementedError('This method is not implemented.')

    def forward_test(self, lq, gt=None, **kwargs):
        outputs = self.forward_dummy(lq)
        result = self._test_post_process(outputs, lq, gt)
        return result

    def train_step(self, data_batch, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def evaluate(self, output, gt):
        """Evaluation function. (Copy from mmedit)

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)
        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt,
                                                               crop_border)
        return eval_result

    def _test_post_process(self, outputs, lq, gt=None):
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(outputs, gt))
        else:
            results = dict(lq=lq.cpu(), output=outputs)
            if gt is not None:
                results['gt'] = gt.cpu()

        return results


class ONNXRuntimeRestorer(DeployBaseRestorer):
    """Wrapper for restorer's inference with ONNXRuntime."""

    def __init__(self, model_file, device_id, test_cfg=None, **kwargs):
        super(ONNXRuntimeRestorer, self).__init__(
            device_id, test_cfg=test_cfg, **kwargs)

        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_dummy(self, lq, *args, **kwargs):
        ort_outputs = self.model({'input': lq})
        # only concern pred_alpha value
        if isinstance(ort_outputs, (tuple, list)):
            ort_outputs = ort_outputs[0]
        return ort_outputs


class TensorRTRestorer(DeployBaseRestorer):

    def __init__(self, trt_file, device_id, test_cfg=None, **kwargs):
        super(TensorRTRestorer, self).__init__(
            device_id, test_cfg=test_cfg, **kwargs)

        from mmdeploy.apis.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom plugins, \
                you may have to build backend ops with TensorRT')
        model = TRTWrapper(trt_file)
        self.model = model

    def forward_dummy(self, img, *args, **kwargs):
        input_data = img.contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            pred = self.model({'input': input_data})['output']
        pred = pred.detach().cpu().numpy()
        return pred


ONNXRUNTIME_RESTORER_MAP = dict(end2end=ONNXRuntimeRestorer)

TENSORRT_RESTORER_MAP = dict(end2end=TensorRTRestorer)

# TODO: Coming Soon
# PPL_RESTORER_MAP = dict(end2end=PPLClassifier)
# NCNN_RESTORER_MAP = dict(end2end=NCNNClassifier)

BACKEND_RESTORER_MAP = {
    Backend.ONNXRUNTIME: ONNXRUNTIME_RESTORER_MAP,
    Backend.TENSORRT: TENSORRT_RESTORER_MAP,
    # TODO: Coming Soon
    # Backend.PPL: PPL_RESTORER_MAP,
    # Backend.NCNN: NCNN_RESTORER_MAP
}


def build_restorer(model_files, backend, model_cfg, device_id):
    model_map = BACKEND_RESTORER_MAP[backend]

    model_type = 'end2end'
    assert model_type in model_map, f'Unsupported model type: {model_type}'
    backend_model_class = model_map[model_type]

    backend_model = backend_model_class(
        model_files[0], device_id=device_id, test_cfg=model_cfg.test_cfg)

    return backend_model


def build_editing_processor(model_files, model_cfg, deploy_cfg, device_id):

    model_cfg = load_config(model_cfg)[0]
    deploy_cfg = load_config(deploy_cfg)[0]

    backend = get_backend(deploy_cfg)

    assert backend in BACKEND_RESTORER_MAP, \
        f'Unsupported backend type: {backend.value}'

    # TODO: Add other tasks
    return build_restorer(model_files, backend, model_cfg, device_id)
