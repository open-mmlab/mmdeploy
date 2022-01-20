# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import mmcv
from mmcv.utils import Registry

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


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
                 deploy_cfg: Union[str, mmcv.Config] = None):
        super().__init__(deploy_cfg=deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
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

    def forward(self, voxels, num_points, coors):
        input_dict = {
            'voxels': voxels,
            'num_points': num_points,
            'coors': coors
        }
        outputs = self.wrapper(input_dict)
        return outputs

    def show_result(self, *args, **kwargs):
        pass


def build_voxel_detection_model(model_files: Sequence[str],
                                model_cfg: Union[str, mmcv.Config],
                                deploy_cfg: Union[str,
                                                  mmcv.Config], device: str):
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
