# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmengine
import torch
from mmagic.structures import DataSample
from mmagic.apis.inferencers.inference_functions import VIDEO_EXTENSIONS, pad_sequence
from mmengine import Config
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement
from torch import nn

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_root_logger, load_config)

__BACKEND_MODEL = Registry('backend_video_super_resolution_models')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of video super resolution.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        model_cfg(mmengine.Config): Input model config object.
        deploy_cfg(str | mmengine.Config):Deployment config file or loaded
            Config object.
        data_preprocessor (BaseDataPreprocessor): The data preprocessor
                of the model. Default to `None`.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: mmengine.Config,
                 deploy_cfg: Union[str, mmengine.Config] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 **kwargs):
        super().__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.test_cfg = model_cfg.test_cfg
        self.device = device
        self._init_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            **kwargs)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str, **kwargs):
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg,
            **kwargs)

    def convert_to_datasample_list(
            self, predictions: DataSample, data_samples: DataSample,
            inputs: Optional[torch.Tensor]) -> List[DataSample]:
        """Add predictions and destructed inputs (if passed) into a list of
        data samples.

        Args:
            predictions (DataSample): The predictions of the model.
            data_samples (DataSample): The data samples loaded from
                dataloader.
            inputs (Optional[torch.Tensor]): The input of model. Defaults to
                None.

        Returns:
            List[EditDataSample]: A list of modified data samples.
        """

        if inputs is not None:
            destructed_input = self.data_preprocessor.destruct(
                inputs, data_samples, 'img')
            data_samples.set_tensor_data({'input': destructed_input})
        # split to list of data samples
        data_samples = data_samples.split()
        predictions = predictions.split()

        for data_sample, pred in zip(data_samples, predictions):
            data_sample.output = pred

        return data_samples

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
        predictions = DataSample(pred_img=outputs.cpu())

        predictions = self.convert_to_datasample_list(predictions,
                                                      data_samples, inputs)

        return predictions


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmagic format."""

    def __init__(self, *args, **kwargs):
        kwargs.update(dict(data_preprocessor=None))
        super(SDKEnd2EndModel, self).__init__(*args, **kwargs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                *args,
                **kwargs) -> list:
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
        for input in inputs:
            output = self.wrapper.invoke(
                input.permute(1, 2, 0).contiguous().detach().cpu().numpy())
            outputs.append(
                torch.from_numpy(output).permute(2, 0, 1).contiguous())
        outputs = torch.stack(outputs, 0)
        outputs = DataSample(pred_img=outputs.cpu()).split()

        for data_sample, pred in zip(data_samples, outputs):
            data_sample.output = pred
        return data_samples


def build_super_resolution_model(
        model_files: Sequence[str],
        model_cfg: Union[str, mmengine.Config],
        deploy_cfg: Union[str, mmengine.Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build super resolution model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model
        data_preprocessor (BaseDataPreprocessor | Config): The data
            preprocessor of the model.
    Returns:
        End2EndModel: Super Resolution model for a configured backend.
    """
    model_cfg = load_config(model_cfg)[0]
    deploy_cfg = load_config(deploy_cfg)[0]

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')
    backend_model = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_model
