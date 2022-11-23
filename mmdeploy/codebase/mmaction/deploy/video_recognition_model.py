# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.utils import Registry

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_video_recognizer', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of segmentation.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        class_names (Sequence[str]): A list of string specifying class names.
        palette (np.ndarray): The palette of segmentation map.
        deploy_cfg (str | mmcv.Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, mmcv.Config] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self._init_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            **kwargs)

    def _init_wrapper(self, backend, backend_files, device, **kwargs):
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg,
            **kwargs)

    def forward(self, imgs: List[torch.Tensor], *args, **kwargs) -> list:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """

        if isinstance(imgs, list):
            input_img = imgs[0].contiguous()
        else:
            input_img = imgs.contiguous()
        outputs = self.forward_test(input_img, *args, **kwargs)

        return list(outputs)

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs) -> \
            List[np.ndarray]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            List[np.ndarray]: A list of classification prediction.
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [out.detach().cpu().numpy().flatten() for out in outputs]
        return outputs

    def show_result(self,
                    img: np.ndarray,
                    result: list,
                    win_name: str = '',
                    palette: Optional[np.ndarray] = None,
                    show: bool = True,
                    opacity: float = 0.5,
                    out_file: str = None):
        """Show predictions of segmentation.
        Args:
            img: (np.ndarray): Input image to draw predictions.
            result (list): A list of predictions.
            win_name (str): The name of visualization window. Default is ''.
            palette (np.ndarray): The palette of segmentation map.
            show (bool): Whether to show plotted image in windows. Defaults to
                `True`.
            opacity: (float): Opacity of painted segmentation map.
                    Defaults to `0.5`.
            out_file (str): Output image file to save drawn predictions.

        Returns:
            np.ndarray: Drawn image, only if not `show` or `out_file`.
        """
        pass


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmaction format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pipeline = kwargs['deploy_cfg']['backend_config']['pipeline']
        for trans in pipeline:
            if trans['type'] == 'SampleFrames':
                self.clip_len = trans['clip_len']
                self.num_clips = trans['num_clips']

    def forward(self, imgs: List[torch.Tensor], *args, **kwargs) -> list:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        imgs = imgs[0]
        mats = []
        for mat in imgs:
            mats.append(mat.contiguous().detach().cpu().numpy())
        mats = [mat[:, :, ::-1] for mat in mats]
        pred = self.wrapper.handle(mats, (self.clip_len, self.num_clips))
        pred = np.array(pred, dtype=np.float32)
        pred = pred[np.argsort(pred[:, 0])][np.newaxis, :, 1]
        return pred


def build_video_recognition_model(model_files: Sequence[str],
                                  model_cfg: Union[str, mmcv.Config],
                                  deploy_cfg: Union[str, mmcv.Config],
                                  device: str, **kwargs):
    """Build video recognition model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Video recognizer for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_video_recognizer = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        device=device,
        deploy_cfg=deploy_cfg,
        **kwargs)

    return backend_video_recognizer
