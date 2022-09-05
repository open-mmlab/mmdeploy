# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor

from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmedit.deploy.mmediting import MMEDIT_TASK
from mmdeploy.utils import Task, get_input_shape


def process_model_config(model_cfg: mmengine.Config,
                         imgs: Union[Sequence[str], Sequence[np.ndarray]],
                         input_shape: Optional[Sequence[int]] = None):
    """Process the model config.

    Args:
        model_cfg (mmengine.Config): The model config.
        imgs (Sequence[str] | Sequence[np.ndarray]): Input image(s), accepted
            data type are List[str], List[np.ndarray].
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Default: None.

    Returns:
        mmengine.Config: the model config after processing.
    """
    config = deepcopy(model_cfg)
    keys_to_remove = ['gt', 'gt_path']
    # MMEdit doesn't support LoadImageFromWebcam.
    # Remove "LoadImageFromFile" and related metakeys.
    load_from_file = isinstance(imgs[0], str)
    is_static_cfg = input_shape is not None
    if not load_from_file:
        config.test_pipeline.pop(0)
        keys_to_remove.append('lq_path')

    # Fix the input shape by 'Resize'
    if is_static_cfg:
        resize = {
            'type': 'Resize',
            'scale': (input_shape[0], input_shape[1]),
            'keys': ['img']
        }
        config.test_pipeline.insert(1, resize)

    for key in keys_to_remove:
        for pipeline in list(config.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                config.test_pipeline.remove(pipeline)
            if 'keys' in pipeline:
                while key in pipeline['keys']:
                    pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    config.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline:
                while key in pipeline['meta_keys']:
                    pipeline['meta_keys'].remove(key)
    return config


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.
    Args:
        model_cfg Config: Input model Config object.
    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmedit import datasets  # noqa
    from mmedit.registry import DATASETS

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


@MMEDIT_TASK.register_module(Task.SUPER_RESOLUTION.value)
class SuperResolution(BaseTask):
    """BaseTask class of super resolution task.

    Args:
        model_cfg (mmengine.Config): Model config file.
        deploy_cfg (mmengine.Config): Deployment config file.
        device (str): A string specifying device type.
    """

    def __init__(self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config,
                 device: str):
        super(SuperResolution, self).__init__(model_cfg, deploy_cfg, device)

    def build_backend_model(self,
                            model_files: Sequence[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files. Default is None.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .super_resolution_model import build_super_resolution_model
        model = build_super_resolution_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            **kwargs)
        return model

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Sequence[int] = None,
                     data_preprocessor: Optional[BaseDataPreprocessor] = None)\
            -> Tuple[Dict, torch.Tensor]:
        """Create input for editing processor.

        Args:
            imgs (str | np.ndarray): Input image(s).
            input_shape (Sequence[int] | None): A list of two integer in
             (width, height) format specifying input shape. Defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmcv.transforms import Compose

        if isinstance(imgs, (list, tuple)):
            if not isinstance(imgs[0], (np.ndarray, str)):
                raise AssertionError('imgs must be strings or numpy arrays')
        elif isinstance(imgs, (np.ndarray, str)):
            imgs = [imgs]
        else:
            raise AssertionError('imgs must be strings or numpy arrays')

        cfg = process_model_config(self.model_cfg, imgs, input_shape)

        test_pipeline = Compose(cfg.test_pipeline)
        data_arr = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                data = dict(img=img)
            else:
                data = dict(img_path=img)

            data = test_pipeline(data)
            data_arr.append(data)
        data = pseudo_collate(data_arr)
        if data_preprocessor is not None:
            data = data_preprocessor(data, False)
            return data, data['inputs']
        else:
            return data, BaseTask.get_tensor_from_input(data)

    def get_visualizer(self, name: str, save_dir: str):
        """Visualize predictions of a model.

        Args:
            name (str): The name of visualization window.
            save_dir (str): The directory to save images.
        """
        from mmedit.utils import register_all_modules
        register_all_modules(init_default_scope=False)
        visualizer = super().get_visualizer(name, save_dir)
        metainfo = _get_dataset_metainfo(self.model_cfg)
        if metainfo is not None:
            visualizer.dataset_meta = metainfo
        return visualizer

    @staticmethod
    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config for mmedit.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any]) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info
            and image tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        return input_data['img']

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.test_pipeline
        for item in preprocess:
            if 'Normalize' == item['type'] and 'std' in item:
                item['std'] = [255, 255, 255]
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Nonthing for super resolution.
        """
        return dict()

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'generator' in self.model_cfg.model, 'generator not in model '
        'config'
        assert 'type' in self.model_cfg.model.generator, 'generator contains '
        'no type'
        name = self.model_cfg.model.generator.type.lower()
        return name
