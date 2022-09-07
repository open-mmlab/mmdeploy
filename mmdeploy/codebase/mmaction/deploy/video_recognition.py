# Copyright (c) OpenMMLab. All rights reserved.

from .mmaction import MMACTION_TASK
import os.path as osp
from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, Any

import numpy as np
import torch
import mmengine
from mmengine.model import BaseDataPreprocessor
from mmengine.dataset import pseudo_collate

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task, get_root_logger
from mmdeploy.utils.config_utils import get_input_shape


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
    logger = get_root_logger()
    cfg = model_cfg.deepcopy()
    test_pipeline_cfg = cfg.test_pipeline
    if 'Init' not in test_pipeline_cfg[0]['type']:
        test_pipeline_cfg = [dict(type='OpenCVInit')] + test_pipeline_cfg
    else:
        test_pipeline_cfg[0] = dict(type='OpenCVInit')
    for i, trans in enumerate(test_pipeline_cfg):
        if 'Decode' in trans['type']:
            test_pipeline_cfg[i] = dict(type='OpenCVDecode')
    cfg.test_pipeline = test_pipeline_cfg

    # check whether input_shape is valid
    if input_shape is not None:
        has_crop = False
        crop_size = -1
        has_resize = False
        scale = (-1, -1)
        keep_ratio = True
        for trans in cfg.test_pipeline:
            if trans['type'] == 'Resize':
                has_resize = True
                keep_ratio = trans.get('keep_ratio', True)
                scale = trans.scale
            if trans['type'] in ['TenCrop', 'CenterCrop', 'ThreeCrop']:
                has_crop = True
                crop_size = trans.crop_size

        if has_crop and tuple(input_shape) != (crop_size, crop_size):
            logger.error(
                f'`input shape` should be equal to `crop_size`: {crop_size},'
                f' but given: {input_shape}')
        if has_resize and (not has_crop):
            if keep_ratio:
                logger.error(
                    'Resize should set `keep_ratio` to False'
                    ' when `input shape` is given.')
            if tuple(input_shape) != scale:
                logger.error(
                    f'`input shape` should be equal to `scale`: {scale},'
                    f' but given: {input_shape}')
    return cfg


@MMACTION_TASK.register_module(Task.VIDEO_RECOGNITION.value)
class VideoRecognition(BaseTask):
    """VideoRecognition task class.

    Args:
        model_cfg (Config): Original PyTorch model config file.
        deploy_cfg (Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config,
                 device: str):
        super(VideoRecognition, self).__init__(model_cfg, deploy_cfg, device)

    def build_backend_model(self,
                            model_files: Sequence[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .video_recognition_model import build_video_recognition_model
        model = build_video_recognition_model(
            model_files, self.model_cfg, self.deploy_cfg, device=self.device)
        model.to(self.device)
        model.eval()
        return model

    def create_input(self,
                     imgs: Union[str, np.ndarray],
                     input_shape: Sequence[int] = None,
                     data_preprocessor: Optional[BaseDataPreprocessor] = None)\
            -> Tuple[Dict, torch.Tensor]:
        """Create input for video recognition.

        Args:
            imgs (str | np.ndarray): Input image(s), accepted data type are
                `str`, `np.ndarray`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        if isinstance(imgs, (list, tuple)):
            if not isinstance(imgs[0], str):
                raise AssertionError('imgs must be strings')
        elif isinstance(imgs, str):
            imgs = [imgs]
        else:
            raise AssertionError('imgs must be strings')

        from mmcv.transforms.wrappers import Compose
        model_cfg = process_model_config(self.model_cfg, imgs,
                                         input_shape)
        test_pipeline = Compose(model_cfg.test_pipeline)

        data = []
        for img in imgs:
            data_ = dict(filename=img, label=-1, start_index=0, modality='RGB')
            data_ = test_pipeline(data_)
            data.append(data_)

        data = pseudo_collate(data)
        if data_preprocessor is not None:
            data = data_preprocessor(data, False)
            return data, data['inputs']
        else:
            return data, BaseTask.get_tensor_from_input(data)

    @staticmethod
    def run_inference(model, model_inputs: Dict[str, torch.Tensor]):
        """Run inference once for a segmentation model of mmseg.

        Args:
            model (nn.Module): Input model.
            model_inputs (dict): A dict containing model inputs tensor and
                meta info.

        Returns:
            list: The predictions of model inference.
        """
        return model(**model_inputs)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
        """Get a certain partition config.

        Args:
            partition_type (str): A string specifying partition type.

        Returns:
            dict: A dictionary of partition config.
        """
        raise NotImplementedError('Not supported yet.')

    @staticmethod
    def get_tensor_from_input(input_data: Dict[str, Any],
                              **kwargs) -> torch.Tensor:
        """Get input tensor from input data.

        Args:
            input_data (dict): Input data containing meta info and image
                tensor.
        Returns:
            torch.Tensor: An image in `Tensor`.
        """
        return input_data['inputs']

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.data.val.pipeline
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.cls_head.type
        return postprocess

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name
