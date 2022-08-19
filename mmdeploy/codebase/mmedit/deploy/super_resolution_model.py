# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.utils import Registry
from mmedit.core import psnr, ssim, tensor2img

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_models', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of super resolution.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        model_cfg(mmcv.Config): Input model config object.
        deploy_cfg(str | mmcv.Config):Deployment config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: mmcv.Config,
                 deploy_cfg: Union[str, mmcv.Config] = None,
                 **kwargs):
        super().__init__(deploy_cfg=deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.test_cfg = model_cfg.test_cfg
        self.allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}
        self._init_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            **kwargs)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str, **kwargs):
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg,
            **kwargs)

    def forward(self,
                lq: torch.Tensor,
                test_mode: bool = False,
                *args,
                **kwargs) -> Union[list, dict]:
        """Run test inference for restorer.

        We want forward() to output an image or a evaluation result.
        When test_mode is set, the output is evaluation result. Otherwise
        it is an image.

        Args:
            lq (torch.Tensor): The input low-quality image of the model.
            test_mode (bool): When test_mode is set, the output is evaluation
                result. Otherwise it is an image. Default to `False`.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list | dict: High resolution image or a evaluation results.
        """

        if test_mode:
            return self.forward_test(lq, *args, **kwargs)
        else:
            return self.forward_dummy(lq, *args, **kwargs)

    def forward_test(self,
                     lq: torch.Tensor,
                     gt: Optional[torch.Tensor] = None,
                     meta: List[Dict] = None,
                     save_path=None,
                     *args,
                     **kwargs):
        """Run inference for restorer to generate evaluation result.

        Args:
            lq (torch.Tensor): The input low-quality image of the model.
            gt (torch.Tensor): The ground truth of input image. Defaults to
                `None`.
            meta (List[Dict]): The meta infomations of MMEditing.
            save_path (str): Path to save image. Default: None.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            dict: Evaluation results.
        """
        outputs = self.forward_dummy(lq)
        result = self.test_post_process(outputs, lq, gt)

        # Align to mmediting BasicRestorer
        if save_path:
            outputs = [torch.from_numpy(i) for i in outputs]

            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            save_path = osp.join(save_path, f'{folder_name}.png')

            mmcv.imwrite(tensor2img(outputs), save_path)

        return result

    def forward_dummy(self, lq: torch.Tensor, *args, **kwargs):
        """Run test inference for restorer with backend wrapper.

        Args:
            lq (torch.Tensor): The input low-quality image of the model.

        Returns:
            list[np.ndarray] : High resolution image.
        """
        outputs = self.wrapper({self.input_name: lq})
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [out.detach().cpu().numpy() for out in outputs]
        return outputs

    def evaluate(self, output: Union[torch.Tensor, np.ndarray],
                 gt: torch.Tensor):
        """Evaluation function implemented in mmedit.

        Args:
            output (torch.Tensor | np.ndarray): Model output with
                shape (n, c, h, w).
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

    def test_post_process(self,
                          outputs: List[np.ndarray],
                          lq: torch.Tensor,
                          gt: Optional[torch.Tensor] = None):
        """Get evaluation results by post-processing model outputs.

        Args:
            output (list[np.ndarray]) : The output high resolution image.
            lq (torch.Tensor): The input low-quality image of the model.
            gt (torch.Tensor): The ground truth of input image, default is
                `None`.

        Returns:
            dict: Evaluation results.
        """
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(outputs[0], gt))
        else:
            results = dict(lq=lq.cpu(), output=outputs)
            if gt is not None:
                results['gt'] = gt.cpu()

        return results

    def show_result(self, *args, **kwargs):
        raise NotImplementedError


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmedit format."""

    def forward(self,
                lq: torch.Tensor,
                gt: Optional[torch.Tensor] = None,
                test_mode: bool = False,
                *args,
                **kwargs) -> Union[list, dict]:
        """Run test inference for restorer.

        We want forward() to output an image or a evaluation result.
        When test_mode is set, the output is evaluation result. Otherwise
        it is an image.

        Args:
            lq (torch.Tensor): The input low-quality image of the model.
            test_mode (bool): When test_mode is set, the output is evaluation
                result. Otherwise it is an image. Default to `False`.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list | dict: High resolution image or a evaluation results.
        """
        img = tensor2img(lq)
        output = self.wrapper.invoke(img)
        if test_mode:
            output = torch.from_numpy(output)
            output = output.permute(2, 0, 1)
            output = output / 255.
            results = self.test_post_process([output], lq, gt)
            return results
        else:
            return [output]


def build_super_resolution_model(model_files: Sequence[str],
                                 model_cfg: Union[str, mmcv.Config],
                                 deploy_cfg: Union[str, mmcv.Config],
                                 device: str, **kwargs):
    model_cfg = load_config(model_cfg)[0]
    deploy_cfg = load_config(deploy_cfg)[0]

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_model = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        device=device,
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg,
        **kwargs)

    return backend_model
