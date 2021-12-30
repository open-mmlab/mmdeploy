# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcls.datasets import DATASETS
from mmcls.models.classifiers.base import BaseClassifier

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import Backend, get_backend, load_config


class End2EndModel(BaseBackendModel):
    """End to end model for inference of classification.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        class_names (Sequence[str]): A list of string specifying class names.
        deploy_cfg (str | mmcv.Config): Deployment config file or loaded Config
            object.
    """

    def __init__(
        self,
        backend: Backend,
        backend_files: Sequence[str],
        device: str,
        class_names: Sequence[str],
        deploy_cfg: Union[str, mmcv.Config] = None,
    ):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)
        self.CLASSES = class_names
        self.deploy_cfg = deploy_cfg
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            output_names=output_names)

    def forward(self, img: List[torch.Tensor], *args, **kwargs) -> list:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """

        if isinstance(img, list):
            input_img = img[0].contiguous()
        else:
            input_img = img.contiguous()
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
        outputs = [out.detach().cpu().numpy() for out in outputs]
        return outputs

    def show_result(self,
                    img: np.ndarray,
                    result: list,
                    win_name: str,
                    show: bool = True,
                    out_file: str = None):
        """Show predictions of classification.
        Args:
            img: (np.ndarray): Input image to draw predictions.
            result (list): A list of predictions.
            win_name (str): The name of visualization window.
            show (bool): Whether to show plotted image in windows. Defaults to
                `True`.
            out_file (str): Output image file to save drawn predictions.

        Returns:
            np.ndarray: Drawn image, only if not `show` or `out_file`.
        """
        return BaseClassifier.show_result(
            self, img, result, show=show, win_name=win_name, out_file=out_file)


def get_classes_from_config(model_cfg: Union[str, mmcv.Config]):
    """Get class name from config.

    Args:
        model_cfg (str | mmcv.Config): Input model config file or
            Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    model_cfg = load_config(model_cfg)[0]
    module_dict = DATASETS.module_dict
    data_cfg = model_cfg.data

    if 'train' in data_cfg:
        module = module_dict[data_cfg.train.type]
    elif 'val' in data_cfg:
        module = module_dict[data_cfg.val.type]
    elif 'test' in data_cfg:
        module = module_dict[data_cfg.test.type]
    else:
        raise RuntimeError(f'No dataset config found in: {model_cfg}')

    return module.CLASSES


def build_classification_model(model_files: Sequence[str],
                               model_cfg: Union[str, mmcv.Config],
                               deploy_cfg: Union[str, mmcv.Config],
                               device: str, **kwargs):
    """Build classification model for different backend.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Classifier for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    class_names = get_classes_from_config(model_cfg)
    backend_classifier = End2EndModel(
        backend,
        model_files,
        device,
        class_names,
        deploy_cfg=deploy_cfg,
        **kwargs)
    return backend_classifier
