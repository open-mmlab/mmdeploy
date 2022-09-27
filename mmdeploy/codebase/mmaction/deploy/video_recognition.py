# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from operator import itemgetter
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task, get_root_logger
from mmdeploy.utils.config_utils import get_input_shape
from .mmaction import MMACTION_TASK


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
                logger.error('Resize should set `keep_ratio` to False'
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

    def build_data_preprocessor(self):
        model_cfg = self.model_cfg
        preprocess_cfg = model_cfg.get('preprocess_cfg', None)
        from mmengine.registry import MODELS
        if preprocess_cfg is not None:
            data_preprocessor = MODELS.build(preprocess_cfg)
        else:
            data_preprocessor = BaseDataPreprocessor()

        return data_preprocessor

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
            if not all(isinstance(img, str) for img in imgs):
                raise AssertionError('imgs must be strings')
        elif isinstance(imgs, str):
            imgs = [imgs]
        else:
            raise AssertionError('imgs must be strings')

        from mmcv.transforms.wrappers import Compose
        model_cfg = process_model_config(self.model_cfg, imgs, input_shape)
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

    def visualize(self,
                  image: str,
                  result: list,
                  output_file: str,
                  window_name: str = '',
                  show_result: bool = False,
                  **kwargs):
        """Visualize predictions of a model.

        Args:
            model (nn.Module): Input model.
            image (str): Input video to draw predictions on.
            result (list): A list of predictions.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
        """
        logger = get_root_logger()
        try:
            import decord
            from moviepy.editor import ImageSequenceClip
        except:
            logger.warn('Please install moviepy and decord to '
                        'enable visualize for mmaction')

        save_dir, save_name = osp.split(output_file)
        video = decord.VideoReader(image)
        frames = [x.asnumpy()[..., ::-1] for x in video]
        pred_scores = result.pred_scores.item.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        top1_item = score_sorted[0]
        short_edge_length = min(frames[0].shape[:2])
        scale = short_edge_length // 224.
        img_scale = min(max(scale, 0.3), 3.0)
        text_cfg = {
            'positions': np.array([(img_scale * 5, ) * 2]).astype(np.int32),
            'font_sizes': int(img_scale * 7),
            'font_families': 'monospace',
            'colors': 'white',
            'bboxes': dict(facecolor='black', alpha=0.5, boxstyle='Round')
        }

        visualizer = self.get_visualizer(window_name, save_dir)
        out_frames = []
        for i, frame in enumerate(frames):
            visualizer.set_image(frame)
            texts = [f'Frame {i} of total {len(frames)} frames']
            texts.append(f'top-1 label: {top1_item[0]}, score: {top1_item[0]}')
            visualizer.draw_texts('\n'.join(texts), **text_cfg)
            drawn_img = visualizer.get_image()
            out_frames.append(drawn_img)
        out_frames = [x[..., ::-1] for x in out_frames]
        video_clips = ImageSequenceClip(out_frames, fps=30)
        output_file = output_file[:output_file.rfind('.')] + '.mp4'
        video_clips.write_videofile(output_file)

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
        preprocess = model_cfg.test_pipeline
        return preprocess

    def get_postprocess(self) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Composed of the postprocess information.
        """
        postprocess = self.model_cfg.model.cls_head
        return postprocess

    def get_model_name(self) -> str:
        """Get the model name.

        Return:
            str: the name of the model.
        """
        assert 'type' in self.model_cfg.model, 'model config contains no type'
        name = self.model_cfg.model.type.lower()
        return name
