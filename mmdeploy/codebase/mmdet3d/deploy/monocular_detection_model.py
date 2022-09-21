# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Union

import mmcv
import torch
from mmcv.utils import Registry

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_monocular_model(cls_name: str, registry: Registry, *args,
                                    **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_monocular_detectors', build_func=__build_backend_monocular_model)


@__BACKEND_MODEL.register_module('end2end')
class MonocularDetectionModel(BaseBackendModel):
    """End to end model for inference of monocular detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        model_cfg (str | mmcv.Config): The model config.
        deploy_cfg (str| mmcv.Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: mmcv.Config,
                 deploy_cfg: Union[str, mmcv.Config] = None):
        super().__init__(deploy_cfg=deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
        self.device = device
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
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
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    def forward(self,
                img: Sequence[torch.Tensor],
                img_metas: Sequence[dict],
                return_loss: bool = False,
                rescale: bool = False):
        """Run forward inference.

        Args:
            points (Sequence[torch.Tensor]): A list contains input pcd(s)
                in [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            img_metas (Sequence[dict]): A list of meta info for image(s).
            return_loss (bool): Consistent with the pytorch model.
                Default = False.
            rescale (bool): Whether to rescale the results.
                Defaults = False.
        Returns:
            list: A list contains predictions.
        """
        input_img = img[0].contiguous()
        cam2img = torch.tensor(
            img_metas[0][0]['cam2img'], device=input_img.device)
        cam2img_inverse = torch.inverse(cam2img)
        outputs = self.wrapper({
            'img': input_img,
            'cam2img': cam2img,
            'cam2img_inverse': cam2img_inverse
        })
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [x.squeeze(0) for x in outputs]
        outputs[0] = img_metas[0][0]['box_type_3d'](
            outputs[0], 9, origin=(0.5, 0.5, 0.5))
        outputs.pop(3)  # pop dir_scores

        from mmdet3d.core import bbox3d2result

        bbox_img = [bbox3d2result(*outputs)]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        return bbox_list

    def show_result(self,
                    data: Dict,
                    result: List,
                    out_dir: str,
                    show=False,
                    score_thr=0.3):
        from mmdet3d.apis import show_result_meshlab
        show_result_meshlab(
            data,
            result,
            out_dir,
            score_thr,
            show=show,
            snapshot=not show,
            task='mono-det')


def build_monocular_detection_model(model_files: Sequence[str],
                                    model_cfg: Union[str, mmcv.Config],
                                    deploy_cfg: Union[str, mmcv.Config],
                                    device: str):
    """Build monocular detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model
    Returns:
        VMonocularDetectionModel: Detector for a configured backend.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_detector = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        device=device,
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg)

    return backend_detector
