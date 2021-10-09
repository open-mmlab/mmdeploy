import warnings
from typing import Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmedit.core import psnr, ssim, tensor2img
from mmedit.models import BaseModel

from mmdeploy.utils.config_utils import Backend, get_backend, load_config


class DeployBaseRestorer(BaseModel):
    """Base Class of Wrapper for restorer's inference.

    Args:
        device_id (int): An integer represents device index.
        test_cfg (mmcv.Config) : The test config in model config, which is used
            in evaluation. Defaults to `None`.
    """

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 device_id: int,
                 test_cfg: Optional[mmcv.Config] = None,
                 **kwargs):
        super(DeployBaseRestorer, self).__init__(**kwargs)
        self.test_cfg = test_cfg
        self.device_id = device_id

    def init_weights(self):
        raise NotImplementedError('This method is not implemented.')

    def forward(self, lq: torch.Tensor, test_mode: bool = False, **kwargs):
        """Run test inference for restorer.

        We want forward() to output an image or a evaluation result.
        When test_mode is set, the output is evaluation result. Otherwise
        it is an image.

        Args:
            lq (torch.Tensor): The input low-quality image of the model.
            test_mode (bool): When test_mode is set, the output is evaluation
                result. Otherwise it is an image. Default to `False`.

        Returns:
            torch.Tensor | dict: High resolution image or a evaluation results.
        """

        if test_mode:
            return self.forward_test(lq, **kwargs)
        else:
            return self.forward_dummy(lq, **kwargs)

    def forward_train(self, imgs, labels):
        raise NotImplementedError('This method is not implemented.')

    def forward_test(self,
                     lq: torch.Tensor,
                     gt: Optional[torch.Tensor] = None,
                     **kwargs):
        """Run inference for restorer to generate evaluation result.

        Args:
            lq (torch.Tensor): The input low-quality image of the model.
            gt (torch.Tensor): The ground truth of input image. Defaults to
                `None`.

        Returns:
            dict: Evaluation results.
        """
        outputs = self.forward_dummy(lq)
        result = self._test_post_process(outputs, lq, gt)
        return result

    def train_step(self, data_batch, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def evaluate(self, output: torch.Tensor, gt: torch.Tensor):
        """Evaluation function implemented in mmedit.

        Args:
            output (torch.Tensor): Model output with shape (n, c, h, w).
            gt (torch.Tensor): GT Tensor with shape (n, c, h, w).

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

    def _test_post_process(self,
                           outputs: torch.Tensor,
                           lq: torch.Tensor,
                           gt: Optional[torch.Tensor] = None):
        """Get evaluation results by post-processing model outputs.

        Args:
            output (torch.Tensor) : The output high resolution image.
            lq (torch.Tensor): The input low-quality image of the model.
            gt (torch.Tensor): The ground truth of input image, default is
                `None`.

        Returns:
            dict: Evaluation results.
        """
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
    """Wrapper for restorer's inference with ONNXRuntime.

    Args:
        model_file (str): The path of an input model file.
        device_id (int): An integer represents device index.
        test_cfg (mmcv.Config) : The test config in model config, which is
            used in evaluation. Defaults to `None`.
    """

    def __init__(self,
                 model_file: str,
                 device_id: int,
                 test_cfg: Optional[mmcv.Config] = None,
                 **kwargs):
        super(ONNXRuntimeRestorer, self).__init__(
            device_id, test_cfg=test_cfg, **kwargs)

        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_dummy(self, lq: torch.Tensor, *args, **kwargs):
        """Run test inference for restorer with ONNXRuntime.

        Args:
            lq (torch.Tensor): The input low-quality image of the model.

        Returns:
            list[np.ndarray] : High resolution image.
        """
        ort_outputs = self.model({'input': lq})
        # only concern pred_alpha value
        if isinstance(ort_outputs, (tuple, list)):
            ort_outputs = ort_outputs[0]
        return ort_outputs


class TensorRTRestorer(DeployBaseRestorer):
    """Wrapper for restorer's inference with TensorRT.

    Args:
        trt_file (str): The path of an input model file.
        device_id (int): An integer represents device index.
        test_cfg (mmcv.Config) : The test config in model config, which is
            used in evaluation.
    """

    def __init__(self,
                 trt_file: str,
                 device_id: int,
                 test_cfg: Optional[mmcv.Config] = None,
                 **kwargs):
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

    def forward_dummy(self, lq: torch.Tensor, *args, **kwargs):
        """Run test inference for restorer with TensorRT.

        Args:
            lq (torch.Tensor): The input low-quality image of the model.

        Returns:
            list[np.ndarray]: High resolution image.
        """
        input_data = lq.contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            pred = self.model({'input': input_data})['output']
        pred = pred.detach().cpu().numpy()
        return pred


class PPLRestorer(DeployBaseRestorer):
    """Wrapper for restorer's inference with ppl.

    Args:
        model_file (str): The path of input model file.
        device_id (int): An integer represents device index.
        test_cfg (mmcv.Config): The test config in model config, which is
            used in evaluation.
    """

    def __init__(self,
                 model_file: str,
                 device_id: int,
                 test_cfg: Optional[mmcv.Config] = None,
                 **kwargs):
        super(PPLRestorer, self).__init__(
            device_id, test_cfg=test_cfg, **kwargs)

        from mmdeploy.apis.ppl import PPLWrapper
        self.model = PPLWrapper(model_file, device_id)

    def forward_dummy(self, lq: torch.Tensor, *args, **kwargs):
        """Run test inference for restorer with PPL.

        Args:
            lq (torch.Tensor): Input low-quality image of the model.

        Returns:
            list[np.ndarray]: High resolution image.
        """
        ppl_outputs = self.model({'input': lq})
        # only concern pred_alpha value
        if isinstance(ppl_outputs, (tuple, list)):
            ppl_outputs = ppl_outputs[0]
        return ppl_outputs


ONNXRUNTIME_RESTORER_MAP = dict(end2end=ONNXRuntimeRestorer)

TENSORRT_RESTORER_MAP = dict(end2end=TensorRTRestorer)

PPL_RESTORER_MAP = dict(end2end=PPLRestorer)

BACKEND_RESTORER_MAP = {
    Backend.ONNXRUNTIME: ONNXRUNTIME_RESTORER_MAP,
    Backend.TENSORRT: TENSORRT_RESTORER_MAP,
    Backend.PPL: PPL_RESTORER_MAP,
}


def build_restorer(model_files: Sequence[str], backend: Backend,
                   model_cfg: Union[str, mmcv.Config], device_id: int):
    """Build restorer for different backend.

    Args:
        model_files (Sequence[str]): Input model file(s).
        backend (Backend): Target backend.
        model_cfg (str | mmcv.Config): Input model config file or config
            object.
        device_id (int): An integer represents device index.

    Returns:
        DeployBaseRestorer: Restorer for a configured backend.
    """
    model_map = BACKEND_RESTORER_MAP[backend]

    model_type = 'end2end'
    assert model_type in model_map, f'Unsupported model type: {model_type}'
    backend_model_class = model_map[model_type]

    backend_model = backend_model_class(
        model_files[0], device_id=device_id, test_cfg=model_cfg.test_cfg)

    return backend_model


def build_editing_processor(model_files: Sequence[str],
                            model_cfg: Union[str, mmcv.Config],
                            deploy_cfg: Union[str,
                                              mmcv.Config], device_id: int):
    """Build editing processor for different backend.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device_id (int): An integer represents device index.

    Returns:
        BaseModel: Editing processor for a configured backend.
    """
    model_cfg = load_config(model_cfg)[0]
    deploy_cfg = load_config(deploy_cfg)[0]

    backend = get_backend(deploy_cfg)

    assert backend in BACKEND_RESTORER_MAP, \
        f'Unsupported backend type: {backend.value}'

    # TODO: Add other tasks
    return build_restorer(model_files, backend, model_cfg, device_id)
