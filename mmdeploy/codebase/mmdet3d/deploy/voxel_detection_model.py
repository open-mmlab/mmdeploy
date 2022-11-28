# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.nn import functional as F

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_voxel_model(cls_name: str, registry: Registry, *args,
                                **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_voxel_detectors', build_func=__build_backend_voxel_model)


@__BACKEND_MODEL.register_module('end2end')
class VoxelDetectionModel(BaseBackendModel):
    """End to end model for inference of 3d voxel detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        model_cfg (str | mmcv.Config): The model config.
        deploy_cfg (str|mmcv.Config): Deployment config file or loaded Config
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
                points: Sequence[torch.Tensor],
                img_metas: Sequence[dict],
                return_loss=False):
        """Run forward inference.

        Args:
            points (Sequence[torch.Tensor]): A list contains input pcd(s)
                in [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            img_metas (Sequence[dict]): A list of meta info for image(s).
            return_loss (Bool): Consistent with the pytorch model.
                Default = False.

        Returns:
            list: A list contains predictions.
        """
        results = []
        for i in range(len(img_metas)):
            voxels, num_points, coors = VoxelDetectionModel.voxelize(
                self.model_cfg, points[i])
            input_dict = {
                'voxels': voxels,
                'num_points': num_points,
                'coors': coors
            }
            outputs = self.wrapper(input_dict)
            outputs = self.wrapper.output_to_list(outputs)
            outputs = [x.squeeze(0) for x in outputs]
            bbox_dim = outputs[0].shape[-1]
            outputs[0] = img_metas[0][0]['box_type_3d'](outputs[0], bbox_dim)
            from mmdet3d.core import bbox3d2result

            result = bbox3d2result(*outputs)
            results.append(result)
        return results

    def show_result(self,
                    data: Dict,
                    result: List,
                    out_dir: str,
                    file_name: str,
                    show=False,
                    snapshot=False,
                    **kwargs):
        from mmcv.parallel import DataContainer as DC
        from mmdet3d.core import show_result
        if isinstance(data['points'][0], DC):
            points = data['points'][0]._data[0][0].numpy()
        elif mmcv.is_list_of(data['points'][0], torch.Tensor):
            points = data['points'][0][0]
        else:
            ValueError(f"Unsupported data type {type(data['points'][0])} "
                       f'for visualization!')
        pred_bboxes = result[0]['boxes_3d']
        pred_labels = result[0]['labels_3d']
        pred_bboxes = pred_bboxes.tensor.cpu().numpy()
        show_result(
            points,
            None,
            pred_bboxes,
            out_dir,
            file_name,
            show=show,
            snapshot=snapshot,
            pred_labels=pred_labels)

    @staticmethod
    def voxelize(model_cfg: Union[str, mmcv.Config], points: torch.Tensor):
        """convert kitti points(N, >=3) to voxels.

        Args:
            model_cfg (str | mmcv.Config): The model config.
            points (torch.Tensor): [N, ndim] float tensor. points[:, :3]
                contain xyz points and points[:, 3:] contain other information
                like reflectivity.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        from mmcv.ops import Voxelization
        model_cfg = load_config(model_cfg)[0]
        if 'voxel_layer' in model_cfg.model.keys():
            voxel_layer = model_cfg.model['voxel_layer']
        elif 'pts_voxel_layer' in model_cfg.model.keys():
            voxel_layer = model_cfg.model['pts_voxel_layer']
        else:
            raise
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


def build_voxel_detection_model(model_files: Sequence[str],
                                model_cfg: Union[str, mmcv.Config],
                                deploy_cfg: Union[str,
                                                  mmcv.Config], device: str):
    """Build 3d voxel object detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model

    Returns:
        VoxelDetectionModel: Detector for a configured backend.
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
