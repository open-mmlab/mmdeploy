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
from mmengine.structures import BaseDataElement
from mmagic.structures import DataSample


from mmdeploy.codebase.base import BaseTask
from mmdeploy.codebase.mmagic.deploy.super_resolution import SuperResolution
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
class VideoSuperResolution(SuperResolution):
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

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
        mode: str = "predict",
        *args,
        **kwargs,
    ) -> list:
        """Run test inference for restorer.

        We want forward() to output an image or a evaluation result.
        When test_mode is set, the output is evaluation result. Otherwise
        it is an image.

        Args:
            inputs (torch.Tensor): A list contains input image(s)
                in [C x H x W] format.
            data_samples (List[BaseDataElement], optional): The data samples.
                Defaults to None.
            mode (str, optional): forward mode, only support `predict`.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list | dict: High resolution image or a evaluation results.
        """
        outputs = []

        if self.extra_parameters["window_size"] > 0:  # sliding window framework
            data = pad_sequence(inputs, self.extra_parameters["window_size"])
            # yapf: disable
            for i in range(0, data.size(1) - 2 * (self.extra_parameters['window_size'] // 2)):  # noqa
                # yapf: enable
                data_i = data[:, i:i +
                                self.extra_parameters['window_size']].to(
                                    self.device)
                outputs.append(
                    self.wrapper.invoke(
                data_i.permute(1, 2, 0).contiguous().detach().cpu().numpy()))
        else:  # recurrent framework
            if self.extra_parameters["max_seq_len"] is None:
                outputs = self.model(inputs=inputs.to(self.device), mode="tensor").cpu()
            else:
                for i in range(0, inputs.size(1), self.extra_parameters["max_seq_len"]):
                    data_i = inputs[:, i : i + self.extra_parameters["max_seq_len"]].to(
                        self.device
                    )
                    outputs.append(
                        self.wrapper.invoke(
                            data_i.permute(1, 2, 0).contiguous().detach().cpu().numpy()
                        )
                    )

        outputs = torch.stack(outputs, 0)
        outputs = DataSample(pred_img=outputs.cpu()).split()

        for data_sample, pred in zip(data_samples, outputs):
            data_sample.output = pred
        return data_samples

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
