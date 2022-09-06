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
        result_list = []
        preprocessed = inputs['voxels']
        input_dict = {
            'voxels': preprocessed['voxels'],
            'num_points': preprocessed['num_points'],
            'coors': preprocessed['coors']
        }
        import pdb
        pdb.set_trace()
        outputs = self.wrapper(input_dict)

        return result_list
        # for i in range(len(img_metas)):
            
        #     voxels, num_points, coors = VoxelDetectionModel.voxelize(
        #         self.model_cfg, points[i])

        #     outputs = self.wrapper(input_dict)
        #     result = VoxelDetectionModel.post_process(self.model_cfg,
        #                                               self.deploy_cfg, outputs,
        #                                               img_metas[i],
        #                                               self.device)[0]
        #     result_list.append(result)
        # return result_list

    def show_result(self,
                    data: Dict,
                    result: List,
                    out_dir: str,
                    file_name: str,
                    show=False,
                    snapshot=False,
                    **kwargs):
        import pdb
        pdb.set_trace()
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
    def post_process(model_cfg: Union[str, Config],
                     deploy_cfg: Union[str, Config],
                     outs: Dict,
                     img_metas: Dict,
                     device: str,
                     rescale=False):
        """model post process.

        Args:
            model_cfg (str | Config): The model config.
            deploy_cfg (str|Config): Deployment config file or loaded
            Config object.
            outs (Dict): Output of model's head.
            img_metas(Dict): Meta info for pcd.
            device (str): A string specifying device type.
            rescale (list[torch.Tensor]): whether th rescale bbox.
        Returns:
            list: A list contains predictions, include bboxes, scores, labels.
        """
        from mmdet3d.structures import bbox3d2result
        from mmdet3d.models.builder import build_head
        model_cfg = load_config(model_cfg)[0]
        deploy_cfg = load_config(deploy_cfg)[0]
        if 'bbox_head' in model_cfg.model.keys():
            head_cfg = dict(**model_cfg.model['bbox_head'])
        elif 'pts_bbox_head' in model_cfg.model.keys():
            head_cfg = dict(**model_cfg.model['pts_bbox_head'])
        else:
            raise NotImplementedError('Not supported model.')
        head_cfg['train_cfg'] = None
        head_cfg['test_cfg'] = model_cfg.model['test_cfg']
        head = build_head(head_cfg)
        if device == 'cpu':
            logger = get_root_logger()
            logger.warning(
                'Don\'t suggest using CPU device. Post process can\'t support.'
            )
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                raise NotImplementedError(
                    'Post process don\'t support device=cpu')
        cls_scores = [outs['scores'].to(device)]
        bbox_preds = [outs['bbox_preds'].to(device)]
        dir_scores = [outs['dir_scores'].to(device)]
        with RewriterContext(
                cfg=deploy_cfg,
                backend=deploy_cfg.backend_config.type,
                opset=deploy_cfg.onnx_config.opset_version):
            bbox_list = head.get_bboxes(
                cls_scores, bbox_preds, dir_scores, img_metas, rescale=False)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
        return bbox_results

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
