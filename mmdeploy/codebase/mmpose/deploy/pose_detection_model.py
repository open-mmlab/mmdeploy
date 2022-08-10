# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine import BaseDataElement, Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = Registry('backend_segmentors')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of pose detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (str | mmengine.Config): Deployment config file or loaded
            Config object.
        deploy_cfg (str | mmengine.Config): Model config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, mmengine.Config] = None,
                 model_cfg: Union[str, mmengine.Config] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        from mmpose.models import builder
        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
        self.device = device
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)
        # create head for decoding heatmap
        self.head = builder.build_head(model_cfg.model.head)

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
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                **kwargs):
        """Run forward inference.

        Args:
            batch_inputs (torch.Tensor): Input image(s) in [N x C x H x W]
                format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        batch_size, _, img_height, img_width = batch_inputs.shape
        batch_inputs = batch_inputs.contiguous().to(self.device)
        batch_outputs = self.wrapper({self.input_name: batch_inputs})
        batch_outputs = self.wrapper.output_to_list(batch_outputs)
        heatmaps = batch_outputs[0]
        results = self.head.decode(heatmaps, data_samples)
        return results


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmcls format."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext_info = self.deploy_cfg.ext_info

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)
        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor
        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        anno_size = self.ext_info.image_size
        aspect_ratio = anno_size[0] / anno_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding

        return center, scale

    def _xywh2xyxy(self, x, y, w, h):
        """convert xywh to x1 y1 x2 y2."""
        return x, y, x + w - 1, y + h - 1

    def forward(self, batch_inputs: List[torch.Tensor], *args,
                **kwargs) -> list:
        """Run forward inference.

        Args:
            batch_inputs (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        image_paths = []
        boxes = np.zeros(shape=(batch_inputs.shape[0], 6))
        bbox_ids = []
        sdk_boxes = []
        for i, img_meta in enumerate(kwargs['img_metas']):
            center, scale = self._xywh2cs(*img_meta['bbox'])
            boxes[i, :2] = center
            boxes[i, 2:4] = scale
            boxes[i, 4] = np.prod(scale * 200.0)
            boxes[i, 5] = img_meta[
                'bbox_score'] if 'bbox_score' in img_meta else 1.0
            sdk_boxes.append(self._xywh2xyxy(*img_meta['bbox']))
            image_paths.append(img_meta['image_file'])
            bbox_ids.append(img_meta['bbox_id'])

        pred = self.wrapper.handle(
            [batch_inputs[0].contiguous().detach().cpu().numpy()],
            [sdk_boxes])[0]

        result = dict(
            preds=pred,
            boxes=boxes,
            image_paths=image_paths,
            bbox_ids=bbox_ids)
        return result


def build_pose_detection_model(
        model_files: Sequence[str],
        model_cfg: Union[str, mmengine.Config],
        deploy_cfg: Union[str, mmengine.Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build object segmentation model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmengine.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmengine.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Pose model for a configured backend.
    """
    from mmpose.models.data_preprocessors import PoseDataPreprocessor

    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')
    if isinstance(data_preprocessor, dict):
        dp = data_preprocessor.copy()
        dp_type = dp.pop('type')
        assert dp_type == 'PoseDataPreprocessor'
        data_preprocessor = PoseDataPreprocessor(**dp)
    backend_pose_model = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_pose_model
