# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.utils import Config, Registry
from mmedit.core import L1Evaluation, psnr, ssim, tensor2img

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_ir_config, load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_models', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of inpainting.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
            (e.g. '.onnx' for ONNX Runtime).
        device (str): A string represents device type.
        model_cfg(Config): Input model config object.
        deploy_cfg(str | Config): Deployment config file or loaded Config
            object.
    """
    _eval_metrics = dict(l1=L1Evaluation, psnr=psnr, ssim=ssim)

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: Config,
                 deploy_cfg: Union[str, Config] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)

        self.input_names = get_ir_config(deploy_cfg).input_names

        self.test_cfg = model_cfg.test_cfg

        # init wrapper
        self.wrapper = self._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=self.input_names,
            output_names=self.output_names,
            deploy_cfg=deploy_cfg,
            **kwargs)

    def forward(self,
                masked_img: torch.Tensor,
                mask: torch.Tensor,
                test_mode: bool = False,
                *args,
                **kwargs) -> Union[list, dict]:
        """Run test inference for inpainting.

        We want forward() to output an image or a evaluation result.
        When test_mode is set, the output is evaluation result. Otherwise
        it is an image.

        Args:
            masked_img (torch.Tensor): Image with hole as input.
            mask (torch.Tensor): Mask as input.
            test_mode (bool, optional): Whether use testing mode.
                Defaults to True.

        Returns:
            list | dict: Inpainted image or a evaluation results.
        """
        if test_mode:
            return self.forward_test(masked_img, mask, *args, **kwargs)

        return self.forward_dummy(masked_img, mask, *args, **kwargs)

    def forward_test(self,
                     masked_img: torch.Tensor,
                     mask: torch.Tensor,
                     save_path=None,
                     *args,
                     **kwargs):
        """Run inference for inpaintor to generate evaluation result.

        Args:
            masked_img (torch.Tensor): Image with hole as input.
            mask (torch.Tensor): Mask as input.
            save_path (str, optional): If given a valid str, the results will
                be saved in this path. Defaults to None.

        Returns:
            dict: Evaluation results.
        """
        outputs = self.forward_dummy(masked_img, mask, *args, **kwargs)
        results = self.test_post_process(outputs, masked_img, mask, *args,
                                         **kwargs)

        if save_path is not None:
            outputs = [torch.from_numpy(i).flip(1) for i in outputs]

            filename, _ = osp.splitext(
                osp.basename(kwargs['meta'][0]['gt_img_path']))
            save_path = osp.join(save_path, f'{filename}.png')
            mmcv.imwrite(tensor2img(outputs, min_max=(-1, 1)), save_path)

        return results

    def forward_dummy(self, masked_img: torch.Tensor, mask: torch.Tensor,
                      *args, **kwargs):
        """Run test inference for inpaintor with backend wrapper.

        Args:
            masked_img (torch.Tensor): Image with hole as input.
            mask (torch.Tensor): Mask as input.

        Returns:
            list[np.ndarray] : Inpainted image.
        """
        inputs = dict(masked_img=masked_img, mask=mask)
        outputs = self.wrapper(inputs)
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [out.detach().cpu().numpy() for out in outputs]
        return outputs

    def evaluate(self, output: Union[torch.Tensor, np.ndarray], masked_img,
                 mask, **kwargs):
        """Evaluation function implemented in mmedit.

        Args:
            output (torch.Tensor | np.ndarray): Model output with
                shape (n, c, h, w).
            masked_img (torch.Tensor): Image with hole as input.
            mask (torch.Tensor): Mask as input.

        Returns:
            dict: Evaluation results.
        """

        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)
        gt_img = kwargs['gt_img'].cpu()

        eval_result = dict()
        data_dict = dict(gt_img=gt_img, fake_img=output, mask=mask.cpu())
        for metric in self.test_cfg.metrics:
            if metric in ['ssim', 'psnr']:
                eval_result[metric] = self._eval_metrics[metric](
                    tensor2img(output, min_max=(-1, 1)),
                    tensor2img(gt_img, min_max=(-1, 1)),
                )
            else:
                eval_result[metric] = self._eval_metrics[metric]()(
                    data_dict).item()

        return eval_result

    def test_post_process(self, outputs: list, masked_img, mask, *args,
                          **kwargs):
        """Get evaluation results by post-processing model outputs.

        Args:
            output (list[np.ndarray]) : The output inpainted image.
            masked_img (torch.Tensor): Image with hole as input.
            mask (torch.Tensor): Mask as input.

        Returns:
            dict: Evaluation results.
        """
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert 'gt_img' in kwargs, (
                'evaluation with metrics must have gt images.')
            results = dict(
                eval_result=self.evaluate(outputs[0], masked_img, mask, *args,
                                          **kwargs))
        else:
            results = dict(masked_img=masked_img, fake_img=outputs)
            if 'gt_img' in kwargs:
                results['gt_img'] = kwargs['gt_img'].cpu()

        return results

    def show_result(self, *args, **kwargs):
        raise NotImplementedError


def build_inpainting_model(model_files: Sequence[str],
                           model_cfg: Union[str, Config],
                           deploy_cfg: Union[str,
                                             Config], device: str, **kwargs):
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
