# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, LabelData
from torch import nn

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_root_logger, load_config)

__BACKEND_MODEL = Registry('backend_classifiers')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of classification.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        class_names (Sequence[str]): A list of string specifying class names.
        deploy_cfg (str | Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, Config] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None):
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)
        self.device = device

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

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict') -> Any:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        if inputs.device != torch.device(self.device):
            get_root_logger().warning(f'expect input device {self.device}'
                                      f' but get {inputs.device}.')
        inputs = inputs.to(self.device)
        cls_score = self.wrapper({self.input_name:
                                  inputs})[self.output_names[0]]

        from mmcls.models.heads.cls_head import ClsHead
        predict = ClsHead._get_predictions(
            None, cls_score, data_samples=data_samples)

        return predict


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmcls format."""

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                *args,
                **kwargs) -> list:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """

        pred = self.wrapper.invoke(inputs[0].permute(
            1, 2, 0).contiguous().detach().cpu().numpy())
        pred = np.array(pred, dtype=np.float32)
        pred[np.argsort(pred[:, 0])][np.newaxis, :, 1]
        pred_label = LabelData()
        # TODO need register metrics calculation in deploy or refactor SDK API
        raise NotImplementedError('Not supported yet.')
        data_samples[0].pred_label = pred_label
        return data_samples


def build_classification_model(
        model_files: Sequence[str],
        model_cfg: Union[str, Config],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build classification model for different backend.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.
        data_preprocessor (BaseDataPreprocessor | Config): The data
            preprocessor of the model.

    Returns:
        BaseBackendModel: Classifier for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_classifier = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_classifier
