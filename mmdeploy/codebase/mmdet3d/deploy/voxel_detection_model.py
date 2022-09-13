# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Sequence, Union, Optional, Tuple

import mmcv
import mmengine
import torch
from mmengine.registry import Registry
from mmengine import Config
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from torch.nn import functional as F
from mmengine.structures import BaseDataElement, InstanceData

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.core import RewriterContext
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_root_logger, load_config)

__BACKEND_MODEL = Registry('backend_voxel_detectors')


@__BACKEND_MODEL.register_module('end2end')
class VoxelDetectionModel(BaseBackendModel):
    """End to end model for inference of 3d voxel detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        model_cfg (str | Config): The model config.
        deploy_cfg (str|Config): Deployment config file or loaded
            Config object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, Config],
                 data_preprocessor: Optional[Union[dict, torch.nn.Module]] = None,
                 **kwargs):
        super().__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
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
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)


    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, Config],
                 data_preprocessor: Optional[Union[dict, torch.nn.Module]] = None,
                 **kwargs):
        super().__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        # self.model_cfg = model_cfg
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
                inputs: dict,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict') -> Any:
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
        preprocessed = inputs['voxels']
        input_dict = {
            'voxels': preprocessed['voxels'].to(self.device),
            'num_points': preprocessed['num_points'].to(self.device),
            'coors': preprocessed['coors'].to(self.device)
        }
        outputs = self.wrapper(input_dict)

        return [outputs]


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
    def postprocess(model_cfg: Union[str, Config],
                     deploy_cfg: Union[str, Config],
                     outs: Dict,
                     metas: Dict):
        """postprocess outputs to datasamples

        Args:
            model_cfg (Union[str, Config]): _description_
            deploy_cfg (Union[str, Config]): _description_
            outs (Dict): _description_
            metas (Dict): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if 'bbox_head' not in model_cfg.model:
            raise NotImplementedError('mmdet3d model bbox_head not found')

        from mmengine.registry import MODELS
        head = MODELS.build(model_cfg.model['bbox_head'])
        
        if 'cls_score' not in outs or 'bbox_pred' not in outs or 'dir_cls_pred' not in outs:
            raise RuntimeError('output tensor not found')
        
        if 'test_cfg' not in model_cfg.model:
            raise RuntimeError('test_cfg not found')
        
        cls_score = outs['cls_score']
        bbox_pred = outs['bbox_pred']
        dir_cls_pred = outs['dir_cls_pred']
        
        batch_input_metas = [
            data_samples.metainfo for data_samples in metas
        ]
        
        import numpy as np
        ccc = np.load('/home/PJLAB/konghuanjun/ccc.npy')
        ddd = cls_score.cpu().numpy()
        diff = ddd - ccc
        print('postprocess cls_scores max {}'.format(diff.max()))
        
        data_instances_3d = head.predict_by_feat(cls_scores=[cls_score], 
                                                 bbox_preds=[bbox_pred], 
                                                 dir_cls_preds=[dir_cls_pred],
                                                 batch_input_metas=batch_input_metas,
                                                 cfg=model_cfg.model.test_cfg)
        data_instances_2d = [
            InstanceData() for _ in range(len(data_instances_3d))
        ]
        
        data_samples = metas
        
        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        
        return data_samples


def build_voxel_detection_model(model_files: Sequence[str],
                                model_cfg: Union[str, Config],
                                deploy_cfg: Union[str, Config],
                                device: str,
                                data_preprocessor: Optional[Union[Config,
                                                                BaseDataPreprocessor]] = None,
                                **kwargs
                                ):
    """Build 3d voxel object detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model
        data_preprocessor (BaseDataPreprocessor | Config): The data
            preprocessor of the model.

    Returns:
        VoxelDetectionModel: Detector for a configured backend.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')
    
    backend_detector = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_detector
