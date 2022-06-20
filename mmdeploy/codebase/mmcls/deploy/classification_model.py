# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.utils import Registry
from mmengine import BaseDataElement, Config
from torch import nn

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_root_logger, load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_classifiers', build_func=__build_backend_model)


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
                      device: str):
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
        if batch_inputs.device != torch.device(self.device):
            get_root_logger().warning(f'expect input device {self.device}'
                                      f' but get {batch_inputs.device}.')
        batch_inputs = batch_inputs.to(self.device)
        cls_score = self.wrapper({self.input_name: batch_inputs})['output']

        from mmcls.models.heads.cls_head import ClsHead
        predict = ClsHead._get_predictions(
            None, cls_score, data_samples=data_samples)

        return predict


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmcls format."""

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

        pred = self.wrapper.invoke(
            [img[0].contiguous().detach().cpu().numpy()])[0]
        pred = np.array(pred, dtype=np.float32)
        return pred[np.argsort(pred[:, 0])][np.newaxis, :, 1]


def get_classes_from_config(model_cfg: Union[str, Config]):
    """Get class name from config.

    Args:
        model_cfg (str | Config): Input model config file or
            Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmcls.datasets import DATASETS

    module_dict = DATASETS.module_dict
    model_cfg = load_config(model_cfg)[0]
    data_cfg = model_cfg.data

    def _get_class_names(dataset_type: str):
        dataset = data_cfg.get(dataset_type, None)
        if (not dataset) or (dataset.type not in module_dict):
            return None

        module = module_dict[dataset.type]
        if module.CLASSES is not None:
            return module.CLASSES
        return module.get_classes(dataset.get('classes', None))

    class_names = None
    for dataset_type in ['val', 'test', 'train']:
        class_names = _get_class_names(dataset_type)
        if class_names is not None:
            break

    if class_names is None:
        logger = get_root_logger()
        logger.warning(f'Use generated class names, because \
            it failed to parse CLASSES from config: {data_cfg}')
        num_classes = model_cfg.model.head.num_classes
        class_names = [str(i) for i in range(num_classes)]
    return class_names


def build_classification_model(model_files: Sequence[str],
                               model_cfg: Union[str, Config],
                               deploy_cfg: Union[str, Config], device: str,
                               **kwargs):
    """Build classification model for different backend.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Classifier for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')
    # class_names = get_classes_from_config(model_cfg)

    backend_classifier = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        device=device,
        deploy_cfg=deploy_cfg,
        **kwargs)

    return backend_classifier
