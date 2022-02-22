# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.nn import functional as F

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_detectors', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class VoxelDetectionModel(BaseBackendModel):

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

    def forward(self, points, img_metas, return_loss=False):
        result_list = []
        for i in range(len(img_metas)):
            voxels, num_points, coors = VoxelDetectionModel.voxelize(
                points[i], self.model_cfg)
            input_dict = {
                'voxels': voxels,
                'num_points': num_points,
                'coors': coors
            }
            outputs = self.wrapper(input_dict)
            result = VoxelDetectionModel.post_process(outputs, img_metas[i],
                                                      self.model_cfg)[0]
            result_list.append(result)
        return result_list

    def show_result(self, *args, **kwargs):
        pass

    @staticmethod
    def voxelize(points, model_cfg):
        if isinstance(model_cfg, str):
            model_cfg = mmcv.Config.fromfile(model_cfg)
        from mmdet3d.ops import Voxelization
        voxel_layer = model_cfg.model['voxel_layer']
        voxel_layer = Voxelization(**voxel_layer)
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @staticmethod
    def post_process(outs, img_metas, model_cfg):
        from mmdet3d.core import bbox3d2result
        from mmdet3d.models.builder import build_head
        head = build_head(
            dict(
                **model_cfg.model['bbox_head'],
                train_cfg=None,
                test_cfg=model_cfg.model['test_cfg']))
        cls_scores = [outs['scores']]
        bbox_preds = [outs['bbox_preds']]
        dir_scores = [outs['dir_scores']]
        bbox_list = head.get_bboxes(
            cls_scores, bbox_preds, dir_scores, img_metas, rescale=True)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results


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
