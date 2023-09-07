# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor
import os.path as osp
import mmcv
from typing import List
from mmengine.dataset import Compose


from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmagic.deploy.mmediting import MMAGIC_TASK
from mmdeploy.utils import Task, get_input_shape, get_root_logger
from mmagic.apis.inferencers.base_mmagic_inferencer import (
    InputsType,
    PredType,
    ResType,
)
from mmagic.apis.inferencers.inference_functions import VIDEO_EXTENSIONS, pad_sequence
import glob
import os
from mmagic.utils import tensor2img
import cv2
from mmengine.utils import ProgressBar
from mmengine.logging import MMLogger
from .super_resolution import process_model_config


@MMAGIC_TASK.register_module(Task.VIDEO_SUPER_RESOLUTION.value)
class VideoSuperResolution(BaseTask):
    """BaseTask class of video super resolution task.

    Args:
        model_cfg (mmengine.Config): Model config file.
        deploy_cfg (mmengine.Config): Deployment config file.
        device (str): A string specifying device type.
    """

    extra_parameters = dict(
        start_idx=0, filename_tmpl="{:08d}.png", window_size=0, max_seq_len=None
    )

    def __init__(
        self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config, device: str
    ):
        super(VideoSuperResolution, self).__init__(model_cfg, deploy_cfg, device)

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
        data_preprocessor = deepcopy(
            self.model_cfg.model.get('data_preprocessor', {}))
        data_preprocessor.setdefault('type', 'mmagic.EditDataPreprocessor')
        model = build_super_resolution_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor,
            **kwargs)
        return model

    def preprocess(self, video: InputsType) -> Dict:
        """Process the inputs into a model-feedable format.

        Args:
            video(InputsType): Video to be restored by models.

        Returns:
            results(InputsType): Results of preprocess.
        """
        # build the data pipeline
        if self.model_cfg.get("demo_pipeline", None):
            test_pipeline = self.model_cfg.demo_pipeline
        elif self.model_cfg.get("test_pipeline", None):
            test_pipeline = self.model_cfg.test_pipeline
        else:
            test_pipeline = self.model_cfg.val_pipeline

        # check if the input is a video
        file_extension = osp.splitext(video)[1]
        if file_extension in VIDEO_EXTENSIONS:
            video_reader = mmcv.VideoReader(video)
            # load the images
            data = dict(img=[], img_path=None, key=video)
            for frame in video_reader:
                data["img"].append(np.flip(frame, axis=2))

            # remove the data loading pipeline
            tmp_pipeline = []
            for pipeline in test_pipeline:
                if pipeline["type"] not in [
                    "GenerateSegmentIndices",
                    "LoadImageFromFile",
                ]:
                    tmp_pipeline.append(pipeline)
            test_pipeline = tmp_pipeline
        else:
            # the first element in the pipeline must be
            # 'GenerateSegmentIndices'
            if test_pipeline[0]["type"] != "GenerateSegmentIndices":
                raise TypeError(
                    "The first element in the pipeline must be "
                    f'"GenerateSegmentIndices", but got '
                    f'"{test_pipeline[0]["type"]}".'
                )

            # specify start_idx and filename_tmpl
            test_pipeline[0]["start_idx"] = self.extra_parameters["start_idx"]
            test_pipeline[0]["filename_tmpl"] = self.extra_parameters["filename_tmpl"]

            # prepare data
            sequence_length = len(glob.glob(osp.join(video, "*")))
            lq_folder = osp.dirname(video)
            key = osp.basename(video)
            data = dict(
                img_path=lq_folder, gt_path="", key=key, sequence_length=sequence_length
            )

        # compose the pipeline
        test_pipeline = Compose(test_pipeline)
        data = test_pipeline(data)
        results = data["inputs"].unsqueeze(0) / 255.0  # in cpu
        data["inputs"] = results
        return data

    def create_input(
        self,
        video: InputsType,
        input_shape: Sequence[int] = None,
        data_preprocessor: Optional[BaseDataPreprocessor] = None,
    ) -> Tuple[Dict, torch.Tensor]:
        """Create input for editing processor.

        Args:
            imgs (str | np.ndarray): Input image(s).
            input_shape (Sequence[int] | None): A list of two integer in
             (width, height) format specifying input shape. Defaults to `None`.
            data_preprocessor (BaseDataPreprocessor): The data preprocessor
                of the model. Default to `None`.

        Returns:
            tuple: (data, img), meta information for the input image and input.
        """
        data = self.preprocess(video)
        return data, BaseTask.get_tensor_from_input(data)

    def visualize(self, preds: PredType, result_out_dir: str = "") -> List[np.ndarray]:
        """Visualize result of a model. mmagic does not have visualizer, so
        write visualize function directly.

        Args:
            model (nn.Module): Input model.
            image (str | np.ndarray): Input image to draw predictions on.
            result (list | np.ndarray): A list of result.
            output_file (str): Output file to save drawn image.
            window_name (str): The name of visualization window. Defaults to
                an empty string.
            show_result (bool): Whether to show result in windows, defaults
                to `False`.
        """

        file_extension = os.path.splitext(result_out_dir)[1]
        mmengine.utils.mkdir_or_exist(osp.dirname(result_out_dir))
        prog_bar = ProgressBar(preds.size(1))
        if file_extension in VIDEO_EXTENSIONS:  # save as video
            h, w = preds.shape[-2:]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(result_out_dir, fourcc, 25, (w, h))
            for i in range(0, preds.size(1)):
                img = tensor2img(preds[:, i, :, :, :])
                video_writer.write(img.astype(np.uint8))
                prog_bar.update()
            cv2.destroyAllWindows()
            video_writer.release()
        else:
            for i in range(
                self.extra_parameters["start_idx"],
                self.extra_parameters["start_idx"] + preds.size(1),
            ):
                output_i = preds[:, i - self.extra_parameters["start_idx"], :, :, :]
                output_i = tensor2img(output_i)
                filename_tmpl = self.extra_parameters["filename_tmpl"]
                save_path_i = f"{result_out_dir}/{filename_tmpl.format(i)}"
                mmcv.imwrite(output_i, save_path_i)
                prog_bar.update()

        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(f"Output video is save at {result_out_dir}.")
        return []

    @staticmethod
    def get_partition_cfg(partition_type: str, **kwargs) -> Dict:
        """Get a certain partition config for mmagic.

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

    def get_preprocess(self, *args, **kwargs) -> Dict:
        """Get the preprocess information for SDK.

        Return:
            dict: Composed of the preprocess information.
        """
        input_shape = get_input_shape(self.deploy_cfg)
        model_cfg = process_model_config(self.model_cfg, [''], input_shape)
        meta_keys = [
            'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'valid_ratio'
        ]
        preprocess = model_cfg.test_pipeline

        preprocess.insert(1, model_cfg.model.data_preprocessor)
        preprocess.insert(2, dict(type='ImageToTensor', keys=['img']))
        transforms = preprocess
        for i, transform in enumerate(transforms):
            if 'keys' in transform and transform['keys'] == ['lq']:
                transform['keys'] = ['img']
            if 'key' in transform and transform['key'] == 'lq':
                transform['key'] = 'img'
            if transform['type'] == 'DataPreprocessor':
                transform['type'] = 'Normalize'
                transform['to_rgb'] = transform.get('to_rgb', False)
            if transform['type'] == 'PackInputs':
                meta_keys += transform[
                    'meta_keys'] if 'meta_keys' in transform else []
                transform['meta_keys'] = list(set(meta_keys))
                transform['keys'] = ['img']
                transforms[i]['type'] = 'Collect'
        return transforms

    def get_postprocess(self, *args, **kwargs) -> Dict:
        """Get the postprocess information for SDK.

        Return:
            dict: Postprocess config for super resolution.
        """
        from mmdeploy.utils import get_task_type
        from mmdeploy.utils.constants import SDK_TASK_MAP as task_map
        task = get_task_type(self.deploy_cfg)
        component = task_map[task]['component']
        post_processor = {'type': component}
        return post_processor

    def get_model_name(self, *args, **kwargs) -> str:
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