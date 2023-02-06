# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.utils import Registry
from numpy import ndarray

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_flow', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of optical flow.

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

    def forward(self, imgs: torch.Tensor, *args,
                **kwargs) -> Sequence[ndarray]:
        """Run test inference for restorer.

        We want forward() to output an image or a evaluation result.
        When test_mode is set, the output is evaluation result. Otherwise
        it is an image.

        Args:
            imgs (torch.Tensor): The input low-quality image of the model.
            test_mode (bool): When test_mode is set, the output is evaluation
                result. Otherwise it is an image. Default to `False`.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list | dict: High resolution image or a evaluation results.
        """
        if isinstance(imgs, list):
            input_img = imgs[0].contiguous()
        else:
            input_img = imgs.contiguous()
        outputs = self.forward_test(input_img, *args, **kwargs)

        return list(outputs)

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Run inference for restorer to generate evaluation result.

        Args:
            imgs (torch.Tensor): The input low-quality image of the model.
            save_path (str): Path to save image. Default: None.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            dict: Evaluation results.
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [out.detach().cpu().numpy() for out in outputs]
        return outputs

    # TODO
    def evaluate(self, output: Union[torch.Tensor, np.ndarray],
                 gt: torch.Tensor):
        """Evaluation function implemented in mmflow.

        Args:
            output (torch.Tensor | np.ndarray): Model output with
                shape (n, c, h, w).
            gt (torch.Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        pass

    def show_result(self, *args, **kwargs):
        raise NotImplementedError


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmflow format."""

    # TODO
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
        pass


def build_mmflow_model(model_files: Sequence[str],
                       model_cfg: Union[str, mmcv.Config],
                       deploy_cfg: Union[str,
                                         mmcv.Config], device: str, **kwargs):
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
