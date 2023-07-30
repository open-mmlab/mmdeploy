# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine import Config
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, InstanceData

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)

__BACKEND_MODEL = Registry('backend_mono_detectors')


@__BACKEND_MODEL.register_module('end2end')
class MonoDetectionModel(BaseBackendModel):
    """End to end model for inference of monocular 3D object detection.

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
            inputs (dict): A dict contains `imgs`
            data_samples (List[BaseDataElement]): A list of meta info for
                image(s).

        Returns:
            list: A list contains predictions.
        """
        preprocessed = inputs['imgs']
        input_dict = {
            'input': preprocessed.to(self.device),
        }
        outputs = self.wrapper(input_dict)
        if data_samples is None:
            return outputs

        prediction = MonoDetectionModel.postprocess(
            model_cfg=self.model_cfg,
            deploy_cfg=self.deploy_cfg,
            outs=outputs,
            metas=data_samples)

        return prediction

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
        if 'cls_score' not in outs or 'bbox_pred' not in outs:  # noqa: E501
            raise RuntimeError('output tensor not found')

        if 'test_cfg' not in model_cfg.model:
            raise RuntimeError('test_cfg not found')

        from mmengine.registry import MODELS
        cls_score = outs['cls_score']
        bbox_pred = outs['bbox_pred']
        batch_input_metas = [data_samples.metainfo for data_samples in metas]

        head = None
        if 'bbox_head' in model_cfg.model:
            head = MODELS.build(model_cfg.model['bbox_head'])
        else:
            raise NotImplementedError('mmdet3d model bbox_head not found')

        if not hasattr(head, 'task_heads'):
            data_instances_3d = head.predict_by_feat(
                cls_scores=[cls_score],
                bbox_preds=[bbox_pred],
                batch_img_metas=batch_input_metas,
            )

            data_samples = MonoDetectionModel.convert_to_datasample(
                data_samples=metas, data_instances_3d=data_instances_3d)
        else:
            raise NotImplementedError('mmdet3d head task_heads not found')

        return data_samples


def build_mono_detection_model(
        model_files: Sequence[str],
        model_cfg: Union[str, Config],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build monocular 3d object detection model for different backends.

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
