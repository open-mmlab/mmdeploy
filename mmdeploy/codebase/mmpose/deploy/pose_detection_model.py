# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import Backend, get_backend, load_config


class End2EndModel(BaseBackendModel):
    """End to end model for inference of pose detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (str | mmcv.Config): Deployment config file or loaded Config
            object.
        deploy_cfg (str | mmcv.Config): Model config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, mmcv.Config] = None,
                 model_cfg: Union[str, mmcv.Config] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)
        from mmpose.models.heads.topdown_heatmap_base_head import \
            TopdownHeatmapBaseHead

        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)
        # create base_head for decoding heatmap
        base_head = TopdownHeatmapBaseHead()
        base_head.test_cfg = model_cfg.model.test_cfg
        self.base_head = base_head

    def _init_wrapper(self, backend, backend_files, device):
        """Initialize backend wrapper.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            output_names=output_names)

    def forward(self, img: torch.Tensor, img_metas: Sequence[Sequence[dict]],
                *args, **kwargs):
        """Run forward inference.

        Args:
            img (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        input_img = img.contiguous()
        outputs = self.forward_test(input_img, img_metas, *args, **kwargs)
        heatmaps = outputs[0]
        key_points = self.base_head.decode(img_metas, heatmaps)
        return key_points

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs) -> \
            List[np.ndarray]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            List[np.ndarray]: A list of segmentation map.
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [out.detach().cpu().numpy() for out in outputs]
        return outputs

    def show_result(self,
                    img: np.ndarray,
                    result: list,
                    win_name: str = '',
                    skeleton: Optional[Sequence[Sequence[int]]] = None,
                    pose_kpt_color: Optional[Sequence[Sequence[int]]] = None,
                    pose_link_color: Optional[Sequence[Sequence[int]]] = None,
                    show: bool = False,
                    out_file: Optional[str] = None,
                    **kwargs):
        """Show predictions of pose.

        Args:
            img: (np.ndarray): Input image to draw predictions.
            result (list): A list of predictions.
            win_name (str): The name of visualization window. Default is ''.
            skeleton (Sequence[Sequence[int]])The connection of keypoints.
                skeleton is 0-based indexing.
            pose_kpt_color (np.array[Nx3]): Color of N keypoints.
                If ``None``, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If ``None``, do not draw links.
            show (bool): Whether to show plotted image in windows.
                Defaults to ``True``.
            out_file (str): Output image file to save drawn predictions.

        Returns:
            np.ndarray: Drawn image, only if not ``show`` or ``out_file``.
        """
        from mmpose.models.detectors import TopDown
        return TopDown.show_result(
            self,
            img,
            result,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            show=show,
            out_file=out_file,
            win_name=win_name)


def build_pose_detection_model(model_files: Sequence[str],
                               model_cfg: Union[str, mmcv.Config],
                               deploy_cfg: Union[str, mmcv.Config],
                               device: str, **kwargs):
    """Build object segmentation model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Pose model for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    backend_pose_model = End2EndModel(
        backend,
        model_files,
        device,
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        **kwargs)

    return backend_pose_model
