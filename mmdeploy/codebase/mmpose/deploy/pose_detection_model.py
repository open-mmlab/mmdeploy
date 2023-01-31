# Copyright (c) OpenMMLab. All rights reserved.
from itertools import zip_longest
from typing import List, Optional, Sequence, Union

import mmengine
import torch
import torch.nn as nn
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, InstanceData

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)

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
        model_cfg (str | mmengine.Config): Model config file or loaded Config
            object.
        data_preprocessor (dict | nn.Module | None): Input data pre-
                processor. Default is ``None``.
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

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str, **kwargs):
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
                data_samples: List[BaseDataElement],
                mode: str = 'predict',
                **kwargs):
        """Run forward inference.

        Args:
            inputs (torch.Tensor): Input image(s) in [N x C x H x W]
                format.
            data_samples (List[BaseDataElement]): A list of meta info for
                image(s).

        Returns:
            list: A list contains predictions.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        inputs = inputs.contiguous().to(self.device)
        batch_outputs = self.wrapper({self.input_name: inputs})
        batch_outputs = self.wrapper.output_to_list(batch_outputs)
        codec = self.model_cfg.codec
        if isinstance(codec, (list, tuple)):
            codec = codec[-1]
        if codec.type == 'SimCCLabel':
            batch_pred_x, batch_pred_y = batch_outputs
            preds = self.head.decode((batch_pred_x, batch_pred_y))
        elif codec.type in ['RegressionLabel', 'IntegralRegressionLabel']:
            preds = self.head.decode(batch_outputs)
        else:
            preds = self.head.decode(batch_outputs[0])
        results = self.pack_result(preds, data_samples)
        return results

    def pack_result(self,
                    preds: Sequence[InstanceData],
                    data_samples: List[BaseDataElement],
                    convert_coordinate: bool = True):
        """Pack pred results to mmpose format
        Args:
            preds (Sequence[InstanceData]): Prediction of keypoints.
            data_samples (List[BaseDataElement]): A list of meta info for
                image(s).
            convert_coordinate (bool): Whether to convert keypoints
                coordinates to original image space. Default is True.
        Returns:
            data_samples (List[BaseDataElement])：
                updated data_samples with predictions.
        """
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
            if convert_coordinate:
                input_size = data_sample.metainfo['input_size']
                bbox_centers = gt_instances.bbox_centers
                bbox_scales = gt_instances.bbox_scales
                keypoints = pred_instances.keypoints
                keypoints = keypoints / input_size * bbox_scales
                keypoints += bbox_centers - 0.5 * bbox_scales
                pred_instances.keypoints = keypoints

            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                data_sample.pred_fields = pred_fields

        return data_samples


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmpose format."""

    def __init__(self, *args, **kwargs):
        kwargs['data_preprocessor'] = None
        super(SDKEnd2EndModel, self).__init__(*args, **kwargs)
        self.ext_info = self.deploy_cfg.ext_info

    def forward(self,
                inputs: List[torch.Tensor],
                data_samples: List[BaseDataElement],
                mode: str = 'predict',
                **kwargs) -> list:
        """Run forward inference.

        Args:
            inputs (List[torch.Tensor]): A list contains input image(s)
                in [H x W x C] format.
            data_samples (List[BaseDataElement]):
                Data samples of image metas.
            mode (str): test mode, only support 'predict'
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        pred_results = []
        for input_img, sample in zip(inputs, data_samples):
            bboxes = sample.gt_instances.bboxes

            # inputs are c,h,w, sdk requested h,w,c
            input_img = input_img.permute(1, 2, 0)
            input_img = input_img.contiguous().detach().cpu().numpy()
            keypoints = self.wrapper.handle(input_img, bboxes.tolist())
            pred = InstanceData(
                keypoints=keypoints[..., :2],
                keypoint_scores=keypoints[..., 2])
            pred_results.append(pred)

        results = self.pack_result(
            pred_results, data_samples, convert_coordinate=False)
        return results


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
        data_preprocessor (Config | BaseDataPreprocessor | None): Input data
            pre-processor. Default is ``None``.
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
