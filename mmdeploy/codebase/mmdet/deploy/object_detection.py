# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Backend, Codebase, Task
from mmdeploy.utils.config_utils import (get_backend, get_input_shape,
                                         is_dynamic_shape)

MMDET_TASK = Registry('mmdet_tasks')


@CODEBASE.register_module(Codebase.MMDET.value)
class MMDetection(MMCodebase):
    """MMDetection codebase class."""

    task_registry = MMDET_TASK

    @classmethod
    def register_deploy_modules(cls):
        """register all rewriters for mmdet."""
        import mmdeploy.codebase.mmdet.models  # noqa: F401
        import mmdeploy.codebase.mmdet.ops
        import mmdeploy.codebase.mmdet.structures  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        """register all related modules and rewriters for mmdet."""
        from mmdet.utils.setup_env import register_all_modules

        cls.register_deploy_modules()
        register_all_modules(True)


def process_model_config(model_cfg: Config,
                         imgs: Union[Sequence[str], Sequence[np.ndarray]],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (Config): The model config.
        imgs (Sequence[str] | Sequence[np.ndarray]): Input image(s), accepted
            data type are List[str], List[np.ndarray].
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        Config: the model config after processing.
    """

    cfg = model_cfg.copy()

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

    pipeline = cfg.test_pipeline

    for i, transform in enumerate(pipeline):
        # for static exporting
        if input_shape is not None:
            if transform.type == 'Resize':
                pipeline[i].keep_ratio = False
                pipeline[i].scale = tuple(input_shape)
            elif transform.type in ('YOLOv5KeepRatioResize', 'LetterResize'):
                pipeline[i].scale = tuple(input_shape)
            elif transform.type == 'Pad' and 'size' in transform:
                pipeline[i].size = tuple(input_shape)

    pipeline = [
        transform for transform in pipeline
        if transform.type != 'LoadAnnotations'
    ]
    cfg.test_pipeline = pipeline
    return cfg


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmdet import datasets  # noqa
    from mmdet.registry import DATASETS

    module_dict = DATASETS.module_dict

    for dataloader_name in [
            'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        dataset_cls = module_dict.get(dataset_cfg.type, None)
        if dataset_cls is None:
            continue
        if hasattr(dataset_cls, '_load_metainfo') and isinstance(
                dataset_cls._load_metainfo, Callable):
            meta = dataset_cls._load_metainfo(
                dataset_cfg.get('metainfo', None))
            if meta is not None:
                return meta
        if hasattr(dataset_cls, 'METAINFO'):
            return dataset_cls.METAINFO

    return None


@MMDET_TASK.register_module(Task.OBJECT_DETECTION.value)
class ObjectDetection(BaseTask):
    """Object Detection task.

    Args:
        model_cfg (Config): The config of the model in mmdet.
        deploy_cfg (Config): The config of deployment.
        device (str): Device name.
    """

    def __init__(self, model_cfg: Config, deploy_cfg: Config,
                 device: str) -> None:
        super().__init__(model_cfg, deploy_cfg, device)

    def build_backend_model(
            self,
            model_files: Optional[str] = None,
            data_preprocessor_updater: Optional[Callable] = None,
            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.
            data_preprocessor_updater (Callable | None): A function to update
                the data_preprocessor. Defaults to None.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .object_detection_model import build_object_detection_model

        data_preprocessor = deepcopy(
            self.model_cfg.model.get('data_preprocessor', {}))
        if data_preprocessor_updater is not None:
            data_preprocessor = data_preprocessor_updater(data_preprocessor)
        data_preprocessor.setdefault('type', 'mmdet.DetDataPreprocessor')

        model = build_object_detection_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor)
        model = model.to(self.device)
        return model.eval()

    def create_input(
        self,
        imgs: Union[str, np.ndarray],
        input_shape: Sequence[int] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """Create input for detector.

        Args:
            imgs (str|np.ndarray): Input image(s), accpeted data type are
                `str`, `np.ndarray`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to `None`.
            data_preprocessor (BaseDataPreprocessor): The data preprocessor
                of the model. Default to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """

        from mmcv.transforms import Compose
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        dynamic_flag = is_dynamic_shape(self.deploy_cfg)
        cfg = process_model_config(self.model_cfg, imgs, input_shape)
        # Drop pad_to_square when static shape. Because static shape should
        # ensure the shape before input image.

        pipeline = cfg.test_pipeline
        if not dynamic_flag:
            transform = pipeline[1]
            if 'transforms' in transform:
                transform_list = transform['transforms']
                for i, step in enumerate(transform_list):
                    if step['type'] == 'Pad' and 'pad_to_square' in step \
                       and step['pad_to_square']:
                        transform_list.pop(i)
                        break
        test_pipeline = Compose(pipeline)
        data = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # TODO: remove img_id.
                data_ = dict(img=img, img_id=0)
            else:
                # TODO: remove img_id.
                data_ = dict(img_path=img, img_id=0)
            # build the data pipeline
            data_ = test_pipeline(data_)
            data.append(data_)

        data = pseudo_collate(data)
        if data_preprocessor is not None:
            data = data_preprocessor(data, False)
            return data, data['inputs']
        else:
            return data, BaseTask.get_tensor_from_input(data)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
        """Get a certain partition config for mmdet.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        from .model_partition_cfg import MMDET_PARTITION_CFG
        assert (partition_type in MMDET_PARTITION_CFG), \
            f'Unknown partition_type {partition_type}'
        return MMDET_PARTITION_CFG[partition_type]

    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        pipeline = model_cfg.test_pipeline
        meta_keys = [
            'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'valid_ratio', 'pad_param'
        ]
        # Extra pad outside datapreprocessor for CenterNet, CornerNet, etc.
        for i, transform in enumerate(pipeline):
            if transform['type'] == 'RandomCenterCropPad':
                if transform['test_pad_mode'][0] == 'logical_or':
                    extra_pad = dict(
                        type='Pad',
                        logical_or_val=transform['test_pad_mode'][1],
                        add_pix_val=transform['test_pad_add_pix'],
                    )
                    pipeline[i] = extra_pad
        transforms = [
            item for item in pipeline if 'Random' not in item['type']
            and 'Annotation' not in item['type']
        ]
        for i, transform in enumerate(transforms):
            # deal with mmyolo
            if transform['type'].startswith('mmdet.'):
                transforms[i]['type'] = transform['type'][6:]
            if 'PackDetInputs' in transform['type']:
                meta_keys += transform[
                    'meta_keys'] if 'meta_keys' in transform else []
                transform['meta_keys'] = list(set(meta_keys))
                transform['keys'] = ['img']
                transforms[i]['type'] = 'Collect'
            if transform['type'] == 'Resize':
                transforms[i]['size'] = transforms[i].pop('scale')
        if self.codebase.value == 'mmyolo':
            transforms = [
                item for item in pipeline
                if item['type'] not in ('ToGray', 'YOLOv5KeepRatioResize')
            ]

        data_preprocessor = model_cfg.model.data_preprocessor

        transforms.insert(-1, dict(type='DefaultFormatBundle'))
        transforms.insert(
            -2,
            dict(
                type='Pad',
                size_divisor=data_preprocessor.get('pad_size_divisor', 1)))
        transforms.insert(
            -3,
            dict(
                type='Normalize',
                to_rgb=data_preprocessor.get('bgr_to_rgb', False),
                mean=data_preprocessor.get('mean', [0, 0, 0]),
                std=data_preprocessor.get('std', [1, 1, 1])))
        return transforms

    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        params = self.model_cfg.model.test_cfg
        type = 'ResizeBBox'  # default for object detection
        if 'rpn' in params:
            params['min_bbox_size'] = params['rpn']['min_bbox_size']
        if 'rcnn' in params:
            params['score_thr'] = params['rcnn']['score_thr']
            if 'mask_thr_binary' in params['rcnn']:
                params['mask_thr_binary'] = params['rcnn']['mask_thr_binary']
        if 'mask_thr_binary' in params:
            type = 'ResizeInstanceMask'  # for instance-seg
            # resize and crop mask to origin image
            params['is_resize_mask'] = True
        if 'mask_thr' in params:
            type = 'ResizeInstanceMask'  # for instance-seg
            # resize and crop mask to origin image
            params['mask_thr_binary'] = params['mask_thr']
            params['is_resize_mask'] = True

        if get_backend(self.deploy_cfg) == Backend.RKNN:
            if 'YOLO' in self.model_cfg.model.type or \
               'RTMDet' in self.model_cfg.model.type:
                bbox_head = self.model_cfg.model.bbox_head
                type = bbox_head.type
                params['anchor_generator'] = bbox_head.get(
                    'anchor_generator', {})
                params['anchor_generator'].update(
                    bbox_head.get('prior_generator', {}))
            else:  # default using base_dense_head
                type = 'BaseDenseHead'
        return dict(type=type, params=params)

    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name

    def get_visualizer(self, name: str, save_dir: str):
        visualizer = super().get_visualizer(name, save_dir)
        metainfo = _get_dataset_metainfo(self.model_cfg)
        if metainfo is not None:
            visualizer.dataset_meta = metainfo
        return visualizer
