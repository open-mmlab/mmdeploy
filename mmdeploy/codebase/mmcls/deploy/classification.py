# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import torch
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, Task, get_root_logger
from mmdeploy.utils.config_utils import get_input_shape

MMCLS_TASK = Registry('mmcls_tasks')


@CODEBASE.register_module(Codebase.MMCLS.value)
class MMClassification(MMCodebase):
    """mmclassification codebase class."""

    task_registry = MMCLS_TASK

    @classmethod
    def register_deploy_modules(cls):
        """register all rewriters for mmcls."""
        import mmdeploy.codebase.mmcls.models  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        """register all related modules and rewriters for mmcls."""
        from mmcls.utils.setup_env import register_all_modules

        cls.register_deploy_modules()
        register_all_modules(True)


def process_model_config(model_cfg: Config,
                         imgs: Union[str, np.ndarray],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (Config): The model config.
        imgs (str | np.ndarray): Input image(s), accepted data type are `str`,
            `np.ndarray`.
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        Config: the model config after processing.
    """
    cfg = model_cfg.deepcopy()
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if isinstance(imgs[0], str):
        if cfg.test_pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.test_pipeline.insert(0, dict(type='LoadImageFromFile'))
    else:
        if cfg.test_pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.test_pipeline.pop(0)
    # check whether input_shape is valid
    if input_shape is not None:
        for pipeline_component in cfg.test_pipeline:
            if 'Crop' in pipeline_component['type']:
                if 'crop_size' in pipeline_component:
                    crop_size = pipeline_component['crop_size']
                    if tuple(input_shape) != (crop_size, crop_size):
                        logger = get_root_logger()
                        logger.warning(
                            f'`input shape` should be equal to `crop_size`: {crop_size},\
                                but given: {input_shape}')
    return cfg


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmcls import datasets  # noqa
    from mmcls.registry import DATASETS

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


@MMCLS_TASK.register_module(Task.CLASSIFICATION.value)
class Classification(BaseTask):
    """Classification task class.

    Args:
        model_cfg (Config): Original PyTorch model config file.
        deploy_cfg (Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: Config, deploy_cfg: Config, device: str):
        super(Classification, self).__init__(model_cfg, deploy_cfg, device)

    def build_data_preprocessor(self):
        """Build data preprocessor.

        Returns:
            nn.Module: A model build with mmcls data_preprocessor.
        """
        model_cfg = deepcopy(self.model_cfg)
        data_preprocessor = deepcopy(model_cfg.get('preprocess_cfg', {}))
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        from mmengine.registry import MODELS
        data_preprocessor = MODELS.build(data_preprocessor)

        return data_preprocessor

    def build_backend_model(
            self,
            model_files: Sequence[str] = None,
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
        from .classification_model import build_classification_model

        data_preprocessor = self.model_cfg.data_preprocessor
        if data_preprocessor_updater is not None:
            data_preprocessor = data_preprocessor_updater(data_preprocessor)
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        model = build_classification_model(
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
        input_shape: Optional[Sequence[int]] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """Create input for classifier.

        Args:
            imgs (Union[str, np.ndarray, Sequence]): Input image(s),
                accepted data type are `str`, `np.ndarray`, Sequence.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Default: None.
            data_preprocessor (BaseDataPreprocessor): The data preprocessor
                of the model. Default to `None`.
        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmengine.dataset import Compose, pseudo_collate
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        assert 'test_pipeline' in self.model_cfg, \
            f'test_pipeline not found in {self.model_cfg}.'
        model_cfg = process_model_config(self.model_cfg, imgs, input_shape)
        pipeline = deepcopy(model_cfg.test_pipeline)
        move_pipeline = []
        while pipeline[-1]['type'] != 'PackClsInputs':
            sub_pipeline = pipeline.pop(-1)
            move_pipeline = [sub_pipeline] + move_pipeline
        pipeline = pipeline[:-1] + move_pipeline + pipeline[-1:]
        pipeline = Compose(pipeline)

        data = []
        for img in imgs:
            # prepare data
            if isinstance(img, str):
                data_ = dict(img_path=img)
            else:
                data_ = dict(img=img)
            # build the data pipeline
            data_ = pipeline(data_)
            data.append(data_)

        data = pseudo_collate(data)
        if data_preprocessor is not None:
            data = data_preprocessor(data, False)
            return data, data['inputs']
        else:
            return data, BaseTask.get_tensor_from_input(data)

    def get_visualizer(self, name: str, save_dir: str):
        """Get mmcls visualizer.

        Args:
            name (str): Name of visualizer.
            save_dir (str): Directory to save drawn results.
        Returns:
            ClsVisualizer: Instance of mmcls visualizer.
        """
        visualizer = super().get_visualizer(name, save_dir)
        metainfo = _get_dataset_metainfo(self.model_cfg)
        if metainfo is not None:
            visualizer.dataset_meta = metainfo
        return visualizer

    def visualize(self,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  **kwargs):
        """Visualize predictions of a model.

        Args:
            model (nn.Module): Input model.
            image (str | np.ndarray): Input image to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
        """
        save_dir, save_name = osp.split(output_file)
        visualizer = self.get_visualizer(window_name, save_dir)

        name = osp.splitext(save_name)[0]
        image = mmcv.imread(image, channel_order='rgb')
        visualizer.add_datasample(
            name, image, result, show=show_result, out_file=output_file)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
        """Get a certain partition config.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError('Not supported yet.')

    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        cfg = process_model_config(self.model_cfg, '', input_shape)
        pipeline = cfg.test_pipeline
        meta_keys = [
            'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'valid_ratio'
        ]
        transforms = [
            item for item in pipeline if 'Random' not in item['type']
        ]
        move_pipeline = []
        import re
        while re.search('Pack[a-z | A-Z]+Inputs',
                        transforms[-1]['type']) is None:
            sub_pipeline = transforms.pop(-1)
            move_pipeline = [sub_pipeline] + move_pipeline
            transforms = transforms[:-1] + move_pipeline + transforms[-1:]
        for i, transform in enumerate(transforms):
            if transform['type'] == 'PackClsInputs':
                meta_keys += transform[
                    'meta_keys'] if 'meta_keys' in transform else []
                transform['meta_keys'] = list(set(meta_keys))
                transform['keys'] = ['img']
                transforms[i]['type'] = 'Collect'
            if transform['type'] == 'Resize':
                transforms[i]['size'] = transforms[i].pop('scale')
            if transform['type'] == 'ResizeEdge':
                transforms[i] = dict(
                    type='Resize',
                    keep_ratio=True,
                    size=(transform['scale'], -1))

        data_preprocessor = self.model_cfg.data_preprocessor
        transforms.insert(-1, dict(type='ImageToTensor', keys=['img']))
        transforms.insert(
            -2,
            dict(
                type='Normalize',
                to_rgb=data_preprocessor.get('to_rgb', False),
                mean=data_preprocessor.get('mean', [0, 0, 0]),
                std=data_preprocessor.get('std', [1, 1, 1])))
        return transforms

    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.model.head
        if 'topk' not in postprocess:
            topk = (1, )
            logger = get_root_logger()
            logger.warning('no topk in postprocess config, using default '
                           'topk value.')
        else:
            topk = postprocess.topk
        postprocess.topk = max(topk)
        return dict(type=postprocess.pop('type'), params=postprocess)

    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'backbone' in self.model_cfg.model, 'backbone not in model '
        'config'
        assert 'type' in self.model_cfg.model.backbone, 'backbone contains '
        'no type'
        name = self.model_cfg.model.backbone.type.lower()
        return name
