# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, Task
from mmdeploy.utils.config_utils import get_input_shape, is_dynamic_shape

MMDET_TASK = Registry('mmdet_tasks')


@CODEBASE.register_module(Codebase.MMDET.value)
class MMDetection(MMCodebase):
    """MMDetection codebase class."""

    task_registry = MMDET_TASK


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
        cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'

    # for static exporting
    if input_shape is not None:
        pipeline = cfg.test_dataloader.dataset.pipeline
        print(f'debugging pipeline: {pipeline}')
        pipeline[1]['scale'] = tuple(input_shape)
        '''
        transforms = pipeline[1]['transforms']
        for trans in transforms:
            trans_type = trans['type']
            if trans_type == 'Resize':
                trans['keep_ratio'] = False
            elif trans_type == 'Pad':
                trans['size_divisor'] = 1
        '''

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

    def __init__(self,
                 model_cfg: Config,
                 deploy_cfg: Config,
                 device: str,
                 experiment_name: str = 'ObjectDetection') -> None:
        super().__init__(model_cfg, deploy_cfg, device, experiment_name)

    def build_backend_model(self,
                            model_files: Optional[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .object_detection_model import build_object_detection_model

        data_preprocessor = deepcopy(
            self.model_cfg.model.get('data_preprocessor', {}))
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

        pipeline = cfg.test_dataloader.dataset.pipeline
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

        data = data[0]
        if data_preprocessor is not None:
            data = data_preprocessor([data], False)
            return data, data[0]
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

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.test_pipeline
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.model.test_cfg
        if 'rpn' in postprocess:
            postprocess['min_bbox_size'] = postprocess['rpn']['min_bbox_size']
        if 'rcnn' in postprocess:
            postprocess['score_thr'] = postprocess['rcnn']['score_thr']
            if 'mask_thr_binary' in postprocess['rcnn']:
                postprocess['mask_thr_binary'] = postprocess['rcnn'][
                    'mask_thr_binary']
        return postprocess

    def get_model_name(self) -> str:
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
