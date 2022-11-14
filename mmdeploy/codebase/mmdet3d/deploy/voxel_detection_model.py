# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Sequence, Union

import mmcv
import torch
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine import Config
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, InstanceData

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)

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
        data_preprocessor (dict|torch.nn.Module): The input preprocessor
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: Union[str, Config],
                 deploy_cfg: Union[str, Config],
                 data_preprocessor: Optional[Union[dict,
                                                   torch.nn.Module]] = None,
                 **kwargs):
        super().__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.model_cfg = model_cfg
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

    def forward(self,
                inputs: dict,
                data_samples: Optional[List[BaseDataElement]] = None,
                **kwargs) -> Any:
        """Run forward inference.

        Args:
            inputs (dict): A dict contains `voxels` which wrapped `voxels`,
                `num_points` and `coors`
            data_samples (List[BaseDataElement]): A list of meta info for
                image(s).

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

        if data_samples is None:
            return outputs

        prediction = VoxelDetectionModel.postprocess(
            model_cfg=self.model_cfg,
            deploy_cfg=self.deploy_cfg,
            outs=outputs,
            metas=data_samples)

        return prediction

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
    def convert_to_datasample(
        data_samples: SampleList,
        data_instances_3d: Optional[List[InstanceData]] = None,
        data_instances_2d: Optional[List[InstanceData]] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

    @staticmethod
    def postprocess(model_cfg: Union[str, Config],
                    deploy_cfg: Union[str, Config], outs: Dict, metas: Dict):
        """postprocess outputs to datasamples.

        Args:
            model_cfg (Union[str, Config]): The model config from
                trainning repo
            deploy_cfg (Union[str, Config]): The deploy config to specify
                backend and input shape
            outs (Dict): output bbox, cls and score
            metas (Dict): DataSample3D for bbox3d render

        Raises:
            NotImplementedError: Only support mmdet3d model with `bbox_head`

        Returns:
            DataSample3D: datatype for render
        """
        if 'cls_score' not in outs or 'bbox_pred' not in outs or 'dir_cls_pred' not in outs:  # noqa: E501
            raise RuntimeError('output tensor not found')

        if 'test_cfg' not in model_cfg.model:
            raise RuntimeError('test_cfg not found')

        from mmengine.registry import MODELS
        cls_score = outs['cls_score']
        bbox_pred = outs['bbox_pred']
        dir_cls_pred = outs['dir_cls_pred']
        batch_input_metas = [data_samples.metainfo for data_samples in metas]

        head = None
        cfg = None
        if 'bbox_head' in model_cfg.model:
            # pointpillars postprocess
            head = MODELS.build(model_cfg.model['bbox_head'])
            cfg = model_cfg.model.test_cfg
        elif 'pts_bbox_head' in model_cfg.model:
            # centerpoint postprocess
            head = MODELS.build(model_cfg.model['pts_bbox_head'])
            cfg = model_cfg.model.test_cfg.pts
        else:
            raise NotImplementedError('mmdet3d model bbox_head not found')

        if not hasattr(head, 'task_heads'):
            data_instances_3d = head.predict_by_feat(
                cls_scores=[cls_score],
                bbox_preds=[bbox_pred],
                dir_cls_preds=[dir_cls_pred],
                batch_input_metas=batch_input_metas,
                cfg=cfg)

            data_samples = VoxelDetectionModel.convert_to_datasample(
                data_samples=metas, data_instances_3d=data_instances_3d)

        else:
            pts = model_cfg.model.test_cfg.pts

            rets = []
            scores_range = [0]
            bbox_range = [0]
            dir_range = [0]
            for i, _ in enumerate(head.task_heads):
                scores_range.append(scores_range[i] + head.num_classes[i])
                bbox_range.append(bbox_range[i] + 8)
                dir_range.append(dir_range[i] + 2)

            for task_id in range(len(head.num_classes)):
                num_class_with_bg = head.num_classes[task_id]

                batch_heatmap = cls_score[:,
                                          scores_range[task_id]:scores_range[
                                              task_id + 1], ...].sigmoid()

                batch_reg = bbox_pred[:,
                                      bbox_range[task_id]:bbox_range[task_id] +
                                      2, ...]
                batch_hei = bbox_pred[:, bbox_range[task_id] +
                                      2:bbox_range[task_id] + 3, ...]

                if head.norm_bbox:
                    batch_dim = torch.exp(bbox_pred[:, bbox_range[task_id] +
                                                    3:bbox_range[task_id] + 6,
                                                    ...])
                else:
                    batch_dim = bbox_pred[:, bbox_range[task_id] +
                                          3:bbox_range[task_id] + 6, ...]

                batch_vel = bbox_pred[:, bbox_range[task_id] +
                                      6:bbox_range[task_id + 1], ...]

                batch_rots = dir_cls_pred[:,
                                          dir_range[task_id]:dir_range[task_id
                                                                       + 1],
                                          ...][:, 0].unsqueeze(1)
                batch_rotc = dir_cls_pred[:,
                                          dir_range[task_id]:dir_range[task_id
                                                                       + 1],
                                          ...][:, 1].unsqueeze(1)

                temp = head.bbox_coder.decode(
                    batch_heatmap,
                    batch_rots,
                    batch_rotc,
                    batch_hei,
                    batch_dim,
                    batch_vel,
                    reg=batch_reg,
                    task_id=task_id)

                assert pts['nms_type'] in ['circle', 'rotate']
                batch_reg_preds = [box['bboxes'] for box in temp]
                batch_cls_preds = [box['scores'] for box in temp]
                batch_cls_labels = [box['labels'] for box in temp]
                if pts['nms_type'] == 'circle':
                    boxes3d = temp[0]['bboxes']
                    scores = temp[0]['scores']
                    labels = temp[0]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    from mmdet3d.models.layers import circle_nms
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            pts['min_radius'][task_id],
                            post_max_size=pts['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task = [ret]
                    rets.append(ret_task)
                else:
                    rets.append(
                        head.get_task_detections(num_class_with_bg,
                                                 batch_cls_preds,
                                                 batch_reg_preds,
                                                 batch_cls_labels,
                                                 batch_input_metas))

            # Merge branches results
            num_samples = len(rets[0])

            ret_list = []
            for i in range(num_samples):
                temp_instances = InstanceData()
                for k in rets[0][i].keys():
                    if k == 'bboxes':
                        bboxes = torch.cat([ret[i][k] for ret in rets])
                        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                        bboxes = batch_input_metas[i]['box_type_3d'](
                            bboxes, head.bbox_coder.code_size)
                    elif k == 'scores':
                        scores = torch.cat([ret[i][k] for ret in rets])
                    elif k == 'labels':
                        flag = 0
                        for j, num_class in enumerate(head.num_classes):
                            rets[j][i][k] += flag
                            flag += num_class
                        labels = torch.cat([ret[i][k].int() for ret in rets])
                temp_instances.bboxes_3d = bboxes
                temp_instances.scores_3d = scores
                temp_instances.labels_3d = labels
                ret_list.append(temp_instances)

            data_samples = VoxelDetectionModel.convert_to_datasample(
                metas, data_instances_3d=ret_list)

        return data_samples


def build_voxel_detection_model(
        model_files: Sequence[str],
        model_cfg: Union[str, Config],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
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
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_detector
