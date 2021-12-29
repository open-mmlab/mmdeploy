# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmcv
import numpy as np
import torch
from mmocr.models.builder import build_head
from mmocr.models.textdet import TextDetectorMixin

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import Backend, get_backend, get_onnx_config, load_config


class End2EndModel(BaseBackendModel):
    """End to end model for inference of text detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
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
        device: str,
        deploy_cfg: Union[str, mmcv.Config] = None,
        model_cfg: Union[str, mmcv.Config] = None,
    ):
        super(End2EndModel, self).__init__()
        model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.show_score = False
        self.bbox_head = build_head(model_cfg.model.bbox_head)
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
        onnx_config = get_onnx_config(self.deploy_cfg)
        output_names = onnx_config['output_names']
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            output_names=output_names)

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
        rescale = kwargs.get('rescale', False)
        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(
                    *(outputs[i].unsqueeze(0)), [img_metas[i]],
                    rescale=rescale) for i in range(len(img_metas))
            ]

        else:
            boundaries = [
                self.bbox_head.get_boundary(
                    *outputs, img_metas, rescale=rescale)
            ]
        return boundaries

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs) -> \
            List[torch.Tensor]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            List[torch.Tensor]: A list of predictions of input images.
        """
        outputs = self.wrapper({'input': imgs})
        outputs = self.wrapper.output_to_list(outputs)
        return outputs

    def show_result(self,
                    img: np.ndarray,
                    result: dict,
                    win_name: str,
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
        return TextDetectorMixin.show_result(
            self,
            img,
            result,
            score_thr=score_thr,
            show=show,
            win_name=win_name,
            out_file=out_file)


class SDKEnd2EndModel(End2EndModel):

    def forward(self, img: Sequence[torch.Tensor],
                img_metas: Sequence[Sequence[dict]], *args, **kwargs) -> list:
        # import pdb; pdb.set_trace()
        boundaries = self.wrapper.invoke(
            [img[0].contiguous().detach().cpu().numpy()])[0]
        boundaries = [list(x) for x in boundaries]
        return [
            dict(
                boundary_result=boundaries, filename=img_metas[0]['filename'])
        ]


def build_text_detection_model(model_files: Sequence[str],
                               model_cfg: Union[str, mmcv.Config],
                               deploy_cfg: Union[str, mmcv.Config],
                               device: str, **kwargs):
    """Build text detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Text detector for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)

    if backend == Backend.SDK:
        model_files.append('TextDetector')
        creator = SDKEnd2EndModel
    else:
        creator = End2EndModel

    backend_text_detector = creator(
        backend,
        model_files,
        device,
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        **kwargs)

    return backend_text_detector
