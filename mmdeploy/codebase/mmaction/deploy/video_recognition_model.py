# Copyright (c) OpenMMLab. All rights reserved.

from typing import Sequence, Union, Optional, List, Any

import torch
import mmengine
from mmengine import Config
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, LabelData
from mmaction.utils import LabelList
from mmdeploy.utils import get_root_logger
from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


__BACKEND_MODEL = Registry('backend_video_recognizer')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of video recognition.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (str | mmengine.Config): Deployment config file or loaded
            Config object.
        model_cfg (str | mmengine.Config): Model config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, Config] = None,
                 model_cfg: Union[str, Config] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)
        model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)
        from mmaction.registry import MODELS
        self.data_preprocessor = MODELS.build(
            model_cfg.model.data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device, **kwargs)
        self.device = device

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
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict') -> Any:
        """Run forward inference.

        Args:
            img (List[torch.Tensor]): A list contains input image(s)
                in [N x num_crops * num_segs x C x H x W] format.
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
        cls_scores = self.wrapper({self.input_name:
                                  inputs})[self.output_names[0]]

        predictions: LabelList = []
        for score in cls_scores:
            label = LabelData(item=score)
            predictions.append(label)
        return predictions


def build_video_recognition_model(model_files: Sequence[str],
                                  model_cfg: Union[str, mmengine.Config],
                                  deploy_cfg: Union[str, mmengine.Config],
                                  device: str,
                                  **kwargs):
    """Build text recognition model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmengine.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmengine.Config): Input deployment config file or
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
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg,
            **kwargs))

    return backend_video_recognizer
