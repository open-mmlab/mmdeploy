# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import List, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.utils import Registry
from mmdet.core import bbox2result
from mmdet.datasets import DATASETS
from mmdet.models import BaseDetector

from mmdeploy.backend.base import get_backend_file_count
from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.codebase.mmdet import get_post_processing_params, multiclass_nms
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_partition_config, load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_detectors', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: mmcv.Config,
                 class_names,
                 deploy_cfg: Union[str, mmcv.Config] = None):
        super().__init__(deploy_cfg=deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.test_cfg = model_cfg.test_cfg
        self.CLASS_NAME = class_names
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    def forward(self, *args, **kwargs):


    def show_result(self, *args, **kwargs):
        pass


def get_classes_from_config(model_cfg):
    model_cfg = load_config(model_cfg)[0]
    if 'class_names' in model_cfg:
        return model_cfg['class_names']
    else:
        raise 'error'


def build_voxel_detection_model(model_files: Sequence[str],
                                 model_cfg: Union[str, mmcv.Config],
                                 deploy_cfg: Union[str, mmcv.Config],
                                 device: str):
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')
    class_names = get_classes_from_config(model_cfg)

    backend_detector = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        class_names=class_names,
        device=device,
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg)

    return backend_detector
