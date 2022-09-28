# Copyright (c) OpenMMLab. All rights reserved.

import copy
import os
from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import mmcv
import mmengine
import numpy as np
import torch
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, Task, get_input_shape, get_root_logger


def process_model_config(
    model_cfg: mmengine.Config,
    imgs: Union[Sequence[str], Sequence[np.ndarray]],
    input_shape: Optional[Sequence[int]] = None,
):
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
    cfg = copy.deepcopy(model_cfg)
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    data_preprocessor = cfg.model.data_preprocessor
    sdk_pipeline = []
    channel_order = 'rgb'
    if input_shape is None:
        codec = cfg.codec
        if isinstance(codec, (list, tuple)):
            codec = codec[0]
        input_shape = np.array(codec['input_size'])

    idx = 0
    while idx < len(test_pipeline):
        trans = test_pipeline[idx]
        if trans.type == 'ToTensor':
            assert idx + 1 < len(test_pipeline) and \
                test_pipeline[idx + 1].type == 'NormalizeTensor'
            trans = test_pipeline[idx + 1]
            trans.type = 'Normalize'
            trans['to_rgb'] = (channel_order == 'rgb')
            trans['mean'] = [x * 255 for x in trans['mean']]
            trans['std'] = [x * 255 for x in trans['std']]
            sdk_pipeline.append(trans)
            sdk_pipeline.append({'type': 'ImageToTensor', 'keys': ['img']})
            idx = idx + 2
            continue

        if trans.type == 'LoadImage':
            if not data_preprocessor.bgr_to_rgb:
                channel_order = 'bgr'
        if trans.type == 'TopDownAffine':
            trans['image_size'] = input_shape
        if trans.type == 'TopDownGetBboxCenterScale':
            trans['image_size'] = input_shape

        sdk_pipeline.append(trans)
        idx = idx + 1
    cfg.test_dataloader.dataset.pipeline = sdk_pipeline
    return cfg


def _get_dataset_metainfo(model_cfg: mmengine.Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.
    Returns:
        (list[str], list[np.ndarray]): Class names and palette
    """
    from mmpose import datasets  # noqa
    from mmpose.registry import DATASETS

    module_dict = DATASETS.module_dict

    for dataloader_name in [
            'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        dataset_mmpose = module_dict.get(dataset_cfg.type, None)
        if dataset_mmpose is None:
            continue
        if hasattr(dataset_mmpose, '_load_metainfo') and isinstance(
                dataset_mmpose._load_metainfo, Callable):
            meta = dataset_mmpose._load_metainfo(
                dataset_cfg.get('metainfo', None))
            if meta is not None:
                return meta
        if hasattr(dataset_mmpose, 'METAINFO'):
            return dataset_mmpose.METAINFO

    return None


MMPOSE_TASK = Registry('mmpose_tasks')


@CODEBASE.register_module(Codebase.MMPOSE.value)
class MMPose(MMCodebase):
    """mmpose codebase class."""
    task_registry = MMPOSE_TASK

    @classmethod
    def register_all_modules(cls):
        from mmpose.utils.setup_env import register_all_modules
        register_all_modules(True)


@MMPOSE_TASK.register_module(Task.POSE_DETECTION.value)
class PoseDetection(BaseTask):
    """Pose detection task class.

    Args:
        model_cfg (mmengine.Config): Original PyTorch model config file.
        deploy_cfg (mmengine.Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config,
                 device: str):
        super().__init__(model_cfg, deploy_cfg, device)

    def build_pytorch_model(self,
                            model_checkpoint: Optional[str] = None,
                            cfg_options: Optional[Dict] = None,
                            **kwargs) -> torch.nn.Module:
        from mmpose.apis import init_model
        from mmpose.utils import register_all_modules
        register_all_modules()
        self.model_cfg.model.test_cfg['flip_test'] = False

        model = init_model(
            self.model_cfg,
            model_checkpoint,
            device=self.device,
            cfg_options=cfg_options)
        return model

    def build_backend_model(self,
                            model_files: Sequence[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files. Default is None.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .pose_detection_model import build_pose_detection_model
        data_preprocessor = self.model_cfg.model.data_preprocessor
        model = build_pose_detection_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor,
            **kwargs)
        return model.eval().to(self.device)

    def create_input(self,
                     imgs: Union[str, np.ndarray, Sequence],
                     input_shape: Sequence[int] = None,
                     data_preprocessor: Optional[BaseDataPreprocessor] = None,
                     **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Create input for pose detection.

        Args:
            imgs (Any): Input image(s), accepted data type are ``str``,
                ``np.ndarray``.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to ``None``.

        Returns:
            tuple: (data, inputs), meta information for the input image
             and input.
        """
        from mmcv.transforms import Compose
        from mmpose.registry import TRANSFORMS
        cfg = self.model_cfg
        if isinstance(imgs, str):
            imgs = mmcv.imread(imgs)
        height, width = imgs.shape[:2]
        # create dummy person results
        person_results = [{'bbox': np.array([0, 0, width, height])}]
        bboxes = np.array([box['bbox'] for box in person_results])

        # build the data pipeline
        test_pipeline = [
            TRANSFORMS.build(c) for c in cfg.test_dataloader.dataset.pipeline
        ]
        test_pipeline = Compose(test_pipeline)
        if input_shape is not None:
            if isinstance(cfg.codec, dict):
                codec = cfg.codec
            elif isinstance(cfg.codec, list):
                codec = cfg.codec[0]
            else:
                raise TypeError(f'Unsupported type {type(cfg.codec)}')
            input_size = codec['input_size']
            if tuple(input_shape) != tuple(input_size):
                logger = get_root_logger()
                logger.warning(f'Input shape from deploy config is not '
                               f'same as input_size in model config:'
                               f'{input_shape} vs {input_size}')

        batch_data = defaultdict(list)
        meta_data = _get_dataset_metainfo(self.model_cfg)
        for bbox in bboxes:
            # prepare data
            bbox_score = np.array([bbox[4] if len(bbox) == 5 else 1
                                   ])  # shape (1,)
            data = {
                'img': imgs,
                'bbox_score': bbox_score,
                'bbox': bbox[None],  # shape (1, 4)
            }
            data.update(meta_data)
            data = test_pipeline(data)
            data['inputs'] = data['inputs'].to(self.device)
            batch_data['inputs'].append(data['inputs'])
            batch_data['data_samples'].append(data['data_samples'])

        if data_preprocessor is not None:
            batch_data = data_preprocessor(batch_data, False)
        input_tensor = batch_data['inputs']
        return batch_data, input_tensor

    def visualize(self,
                  image: Union[str, np.ndarray],
                  result: list,
                  output_file: str,
                  window_name: str,
                  show_result: bool = False,
                  **kwargs):
        """Visualize predictions of a model.

        Args:
            image (str | np.ndarray): Input image to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
        """
        from mmpose.apis.inference import dataset_meta_from_config
        from mmpose.visualization import PoseLocalVisualizer

        save_dir, filename = os.path.split(output_file)
        name = os.path.splitext(filename)[0]
        dataset_meta = dataset_meta_from_config(
            self.model_cfg, dataset_mode='test')
        visualizer = PoseLocalVisualizer(name=name, save_dir=save_dir)
        visualizer.set_dataset_meta(dataset_meta)

        if isinstance(image, str):
            image = mmcv.imread(image, channel_order='rgb')
        visualizer.add_datasample(
            name,
            image,
            data_sample=result,
            draw_gt=False,
            show=show_result,
            out_file=output_file)

    def get_model_name(self, *args, **kwargs) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name

    @staticmethod
    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config for mmpose.

        Args:
            partition_type (str): A string specifying partition type.
        """
        raise NotImplementedError('Not supported yet.')

    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        # TODO: make it work with sdk
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.test_dataloader.dataset.pipeline
        preprocess[0].type = 'LoadImageFromFile'
        return preprocess

    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get the postprocess information for SDK."""
        postprocess = {'type': 'UNKNOWN'}
        if self.model_cfg.model.type == 'TopDown':
            postprocess[
                'type'] = self.model_cfg.model.keypoint_head.type + 'Decode'
            postprocess.update(self.model_cfg.model.test_cfg)
        return postprocess
