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
from mmdeploy.utils import Codebase, Task, get_input_shape, get_root_logger


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
    cfg = deepcopy(model_cfg)

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.test_pipeline[0].type = 'LoadImageFromWebcam'

    # remove some training related pipeline
    removed_indices = []
    for i in range(len(cfg.test_pipeline)):
        if cfg.test_pipeline[i]['type'] in ['LoadAnnotations']:
            removed_indices.append(i)
    for i in reversed(removed_indices):
        cfg.test_pipeline.pop(i)

    # for static exporting
    if input_shape is not None:
        found_resize = False
        for i in range(len(cfg.test_pipeline)):
            if 'Resize' == cfg.test_pipeline[i]['type']:
                cfg.test_pipeline[i]['scale'] = tuple(input_shape)
                cfg.test_pipeline[i]['keep_ratio'] = False
                found_resize = True
        if not found_resize:
            logger = get_root_logger()
            logger.warning(
                f'Not found Resize in test_pipeline: {cfg.test_pipeline}')

    return cfg


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.
    Returns:
        (list[str], list[np.ndarray]): Class names and palette
    """
    from mmseg import datasets  # noqa
    from mmseg.registry import DATASETS

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


MMSEG_TASK = Registry('mmseg_tasks')


@CODEBASE.register_module(Codebase.MMSEG.value)
class MMSegmentation(MMCodebase):
    """mmsegmentation codebase class."""
    task_registry = MMSEG_TASK


@MMSEG_TASK.register_module(Task.SEGMENTATION.value)
class Segmentation(BaseTask):
    """Segmentation task class.

    Args:
        model_cfg (Config): Original PyTorch model config file.
        deploy_cfg (Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: Config, deploy_cfg: Config, device: str):
        super(Segmentation, self).__init__(model_cfg, deploy_cfg, device)

    def build_backend_model(self,
                            model_files: Optional[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .segmentation_model import build_segmentation_model

        data_preprocessor = self.model_cfg.model.data_preprocessor
        model = build_segmentation_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor)
        model = model.to(self.device).eval()
        return model

    def create_input(
        self,
        imgs: Union[str, np.ndarray],
        input_shape: Sequence[int] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None
    ) -> Tuple[Dict, torch.Tensor]:
        """Create input for segmentor.

        Args:
            imgs (Any): Input image(s), accepted data type are `str`,
                `np.ndarray`, `torch.Tensor`.
            input_shape (list[int]): A list of two integer in (width, height)
                format specifying input shape. Defaults to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        from mmseg.datasets.pipelines import Compose
        if isinstance(imgs, str):
            data = dict(img_path=imgs)
        else:
            data = dict(img=imgs)
        cfg = process_model_config(self.model_cfg, imgs, input_shape)
        test_pipeline = Compose(cfg.test_pipeline)
        data = test_pipeline(data)
        if data_preprocessor is not None:
            data = data_preprocessor([data], False)
            input_tensor = data[0]
        else:
            input_tensor = BaseTask.get_tensor_from_input(data)
        return data, input_tensor

    def get_visualizer(self, name: str, save_dir: str):
        """

        Args:
            name:
            save_dir:

        Returns:

        """
        # import to make SegLocalVisualizer could be built
        from mmseg.core.visualization import \
            SegLocalVisualizer  # noqa: F401,F403
        visualizer = super().get_visualizer(name, save_dir)
        # force to change save_dir instead of
        # save_dir/vis_data/vis_image/xx.jpg
        visualizer._vis_backends['LocalVisBackend']._save_dir = save_dir
        visualizer._vis_backends['LocalVisBackend']._img_save_dir = '.'
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
                  opacity: float = 0.5,
                  **kwargs):
        """Visualize segmentation predictions.

        Args:
            image (str | np.ndarray): Input image to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
            opacity: (float): Opacity of painted segmentation map.
                    Defaults to `0.5`.
        """
        save_dir, filename = osp.split(output_file)
        visualizer = self.get_visualizer(window_name, save_dir)
        name = osp.splitext(filename)[0]
        if isinstance(image, str):
            image = mmcv.imread(image, channel_order='rgb')
        visualizer.add_datasample(
            name, image, pred_sample=result.cpu(), show=show_result)

    @staticmethod
    def get_partition_cfg(partition_type: str) -> Dict:
        raise NotImplementedError('Not supported yet.')

    def get_preprocess(self) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        load_from_file = self.model_cfg.test_pipeline[0]
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        preprocess = model_cfg.test_pipeline
        preprocess[0] = load_from_file
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Nonthing for super resolution.
        """
        postprocess = self.model_cfg.model.decode_head
        return postprocess

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'decode_head' in self.model_cfg.model, 'model config contains'
        ' no decode_head'
        if isinstance(self.model_cfg.model.decode_head, list):
            name = self.model_cfg.model.decode_head[-1].type[:-4].lower()
        elif 'type' in self.model_cfg.model.decode_head:
            name = self.model_cfg.model.decode_head.type[:-4].lower()
        else:
            name = 'mmseg_model'
        return name
