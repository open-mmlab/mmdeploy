# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement

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
            backend=backend,
            backend_files=backend_files,
            device=device,
            **kwargs)
        # create head for decoding heatmap
        self.head = builder.build_head(model_cfg.model.head)

    def _init_wrapper(self, backend, backend_files, device, **kwargs):
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
            deploy_cfg=self.deploy_cfg,
            **kwargs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]],
                mode: str = 'predict',
                **kwargs):
        """Run forward inference.

        Args:
            inputs (torch.Tensor): Input image(s) in [N x C x H x W]
                format.
            data_samples (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        inputs = inputs.contiguous().to(self.device)
        batch_outputs = self.wrapper({self.input_name: inputs})
        batch_outputs = self.wrapper.output_to_list(batch_outputs)
        batch_heatmaps = batch_outputs[0]
        # flip test
        test_cfg = self.model_cfg.model.test_cfg
        if test_cfg.get('flip_test', False):
            from mmpose.models.utils.tta import flip_heatmaps
            batch_inputs_flip = inputs.flip(-1).contiguous()
            batch_outputs_flip = self.wrapper(
                {self.input_name: batch_inputs_flip})
            batch_heatmaps_flip = self.wrapper.output_to_list(
                batch_outputs_flip)[0]
            flip_indices = data_samples[0].metainfo['flip_indices']
            batch_heatmaps_flip = flip_heatmaps(
                batch_heatmaps_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (batch_heatmaps + batch_heatmaps_flip) * 0.5
        results = self.pack_result(batch_heatmaps, data_samples)
        return results

    def pack_result(self, heatmaps, data_samples):
        preds = self.head.decode(heatmaps)
        if isinstance(preds, tuple):
            batch_pred_instances, batch_pred_fields = preds
        else:
            batch_pred_instances = preds
            batch_pred_fields = None
        assert len(batch_pred_instances) == len(data_samples)
        if batch_pred_fields is None:
            batch_pred_fields = []

        for pred_instances, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            bbox_centers = gt_instances.bbox_centers
            bbox_scales = gt_instances.bbox_scales
            input_size = data_sample.metainfo['input_size']

            pred_instances.keypoints = pred_instances.keypoints / input_size \
                * bbox_scales + bbox_centers - 0.5 * bbox_scales

            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                data_sample.pred_fields = pred_fields

        return data_samples


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmcls format."""

    def __init__(self, *args, **kwargs):
        kwargs['data_preprocessor'] = None
        super(SDKEnd2EndModel, self).__init__(*args, **kwargs)
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

    def forward(self, inputs: List[torch.Tensor], *args, **kwargs) -> list:
        """Run forward inference.

        Args:
            inputs (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        image_paths = []
        boxes = np.zeros(shape=(inputs.shape[0], 6))
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
            [inputs[0].contiguous().detach().cpu().numpy()], sdk_boxes)

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
