# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.utils import Registry
from mmrotate.models.detectors import RotatedBaseDetector

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.codebase.mmdet.deploy.object_detection_model import \
    get_classes_from_config
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_rotated_detectors', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of rotated detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        class_names (Sequence[str]): A list of string specifying class names.
        device (str): A string represents device type.
        deploy_cfg (str | mmcv.Config): Deployment config file or loaded Config
            object.
        model_cfg (str | mmcv.Config): Model config file or loaded Config
            object.
    """

    def __init__(
        self,
        backend: Backend,
        backend_files: Sequence[str],
        class_names: Sequence[str],
        device: str,
        deploy_cfg: Union[str, mmcv.Config] = None,
        model_cfg: Union[str, mmcv.Config] = None,
    ):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)
        model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)
        self.CLASSES = class_names
        self.deploy_cfg = deploy_cfg
        self.device = device
        self.show_score = False
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        """Initialize the wrapper of backends.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. .onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string represents device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    def forward(self, img: Sequence[torch.Tensor],
                img_metas: Sequence[Sequence[dict]], *args, **kwargs) -> list:
        """Run forward inference.

        Args:
            img (Sequence[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).

        Returns:
            list: A list contains predictions.
        """
        input_img = img[0].contiguous()
        img_metas = img_metas[0]
        outputs = self.forward_test(input_img, img_metas, *args, **kwargs)
        batch_dets, batch_labels = outputs[:2]
        batch_size = input_img.shape[0]
        rescale = kwargs.get('rescale', False)

        results = []

        for i in range(batch_size):
            dets, labels = batch_dets[i], batch_labels[i]
            if rescale:
                scale_factor = img_metas[i]['scale_factor']

                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    assert len(scale_factor) == 4
                    scale_factor = np.array(scale_factor)[None, :]  # [1,4]
                scale_factor = torch.from_numpy(scale_factor).to(
                    device=dets.device)
                dets[:, :4] /= scale_factor
            dets = dets.cpu().numpy()
            labels = labels.cpu().numpy()
            dets_results = [
                dets[labels == i, :] for i in range(len(self.CLASSES))
            ]
            results.append(dets_results)

        return results

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs) -> \
            List[torch.Tensor]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            List[torch.Tensor]: A list of predictions of input images.
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        return outputs

    def show_result(self,
                    img: np.ndarray,
                    result: dict,
                    win_name: str = '',
                    show: bool = True,
                    score_thr: float = 0.3,
                    out_file: str = None):
        """Show predictions of segmentation.
        Args:
            img: (np.ndarray): Input image to draw predictions.
            result (dict): A dict of predictions.
            win_name (str): The name of visualization window.
            show (bool): Whether to show plotted image in windows. Defaults to
                `True`.
            score_thr: (float): The thresh of score. Defaults to `0.3`.
            out_file (str): Output image file to save drawn predictions.

        Returns:
            np.ndarray: Drawn image, only if not `show` or `out_file`.
        """
        return RotatedBaseDetector.show_result(
            self,
            img,
            result,
            score_thr=score_thr,
            show=show,
            win_name=win_name,
            out_file=out_file)


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmrotate format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img: List[torch.Tensor], *args, **kwargs) -> list:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        results = []
        dets, labels = self.wrapper.invoke(
            [img[0].contiguous().detach().cpu().numpy()])[0]
        dets_results = [dets[labels == i, :] for i in range(len(self.CLASSES))]
        results.append(dets_results)

        return results


def build_rotated_detection_model(model_files: Sequence[str],
                                  model_cfg: Union[str, mmcv.Config],
                                  deploy_cfg: Union[str, mmcv.Config],
                                  device: str, **kwargs):
    """Build rotated detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Rotated detector for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    class_names = get_classes_from_config(model_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_rotated_detector = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        class_names=class_names,
        device=device,
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        **kwargs)

    return backend_rotated_detector
