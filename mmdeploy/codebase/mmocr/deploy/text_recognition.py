# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.dist import cast_data_device
from mmengine.model import BaseDataPreprocessor

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task, get_input_shape
from .mmocr import MMOCR_TASK


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
    if isinstance(imgs[0], np.ndarray):
        # set loading pipeline type
        model_cfg.test_pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline = model_cfg._cfg_dict.test_pipeline
    for i, transform in enumerate(test_pipeline):
        if transform.type == 'PackTextRecogInputs':
            test_pipeline[i].meta_keys = tuple(
                j for j in test_pipeline[i].meta_keys if j != 'instances')

        # for static exporting
        if input_shape is not None and transform.type == 'RescaleToHeight':
            resize = {
                'height': input_shape[1],
                'min_width': input_shape[0],
                'max_width': input_shape[0]
            }
            test_pipeline[i].update(resize)

    test_pipeline = [
        transform for transform in test_pipeline
        if transform.type != 'LoadOCRAnnotations'
    ]

    model_cfg.test_pipeline = test_pipeline
    return model_cfg


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.
    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmocr import datasets  # noqa
    from mmocr.registry import DATASETS

    module_dict = DATASETS.module_dict

    for dataloader_name in [
            'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        if isinstance(dataloader_cfg, list):
            dataloader_cfg = dataloader_cfg[0]
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


@MMOCR_TASK.register_module(Task.TEXT_RECOGNITION.value)
class TextRecognition(BaseTask):
    """Text detection task class.

    Args:
        model_cfg (mmengine.Config): Original PyTorch model config file.
        deploy_cfg (mmengine.Config):  Loaded deployment config object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config,
                 device: str):
        super(TextRecognition, self).__init__(model_cfg, deploy_cfg, device)

    def build_backend_model(self,
                            model_files: Optional[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .text_recognition_model import build_text_recognition_model
        model = build_text_recognition_model(
            model_files, self.model_cfg, self.deploy_cfg, device=self.device)
        return model.eval()

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Sequence[int] = None,
                     data_preprocessor: Optional[BaseDataPreprocessor] = None)\
            -> Tuple[Dict, torch.Tensor]:
        """Create input for segmentor.

        Args:
            imgs (str | np.ndarray): Input image(s), accepted data type are
                `str`, `np.ndarray`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        if isinstance(imgs, (list, tuple)):
            if not isinstance(imgs[0], (np.ndarray, str)):
                raise AssertionError('imgs must be strings or numpy arrays')

        elif isinstance(imgs, (np.ndarray, str)):
            imgs = [imgs]
        else:
            raise AssertionError('imgs must be strings or numpy arrays')

        from mmcv.transforms.wrappers import Compose

        # from mmocr.datasets import build_dataset  # noqa: F401
        self.model_cfg = process_model_config(self.model_cfg, imgs,
                                              input_shape)
        test_pipeline = Compose(self.model_cfg.test_pipeline)

        data = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # TODO: remove img_id.
                data_ = dict(
                    img=img, img_id=0, ori_shape=input_shape, instances=None)
            else:
                # TODO: remove img_id.
                data_ = dict(img_path=img, img_id=0)
            # build the data pipeline
            data_ = test_pipeline(data_)
            data.append(data_)

        data = pseudo_collate(data)
        data['inputs'] = cast_data_device(data['inputs'],
                                          torch.device(self.device))
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
        from mmocr.utils import register_all_modules
        register_all_modules(init_default_scope=False)
        visualizer = super().get_visualizer(name, save_dir)
        metainfo = _get_dataset_metainfo(self.model_cfg)
        if metainfo is not None:
            visualizer.dataset_meta = metainfo
        return visualizer

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
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        pipeline = model_cfg.test_dataloader.dataset.pipeline
        meta_keys = [
            'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'valid_ratio'
        ]
        transforms = [
            item for item in pipeline if 'Random' not in item['type']
            and 'Annotation' not in item['type']
        ]
        for i, transform in enumerate(transforms):
            if transform['type'] == 'PackTextRecogInputs':
                meta_keys += transform[
                    'meta_keys'] if 'meta_keys' in transform else []
                transform['meta_keys'] = list(set(meta_keys))
                transform['keys'] = ['img']
                transforms[i]['type'] = 'Collect'
            if transform['type'] == 'Resize':
                transforms[i]['size'] = transforms[i].pop('scale')

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

    def get_postprocess(self,
                        work_dir: Optional[str] = None,
                        **kwargs) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            Dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.model.decoder.postprocessor
        if postprocess.type == 'CTCPostProcessor':
            postprocess.type = 'CTCConvertor'
        if postprocess.type == 'AttentionPostprocessor':
            postprocess.type = 'AttnConvertor'
        import shutil
        shutil.copy(self.model_cfg.dictionary.dict_file,
                    f'{work_dir}/dict_file.txt')
        params = deepcopy(self.model_cfg.dictionary)
        params.update(dict(dict_file='dict_file.txt'))
        postprocess['params'] = params
        return postprocess

    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name
