# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

import mmcv
import torch

from mmdeploy.utils import SDK_TASK_MAP, Backend, get_ir_config, get_task_type


class BaseBackendModel(torch.nn.Module, metaclass=ABCMeta):
    """A backend model wraps the details to initialize and run a backend
    engine."""

    def __init__(self,
                 deploy_cfg: Optional[Union[str, mmcv.Config]] = None,
                 *args,
                 **kwargs):
        """The default for building the base class.

        Args:
            deploy_cfg (str | mmcv.Config | None): The deploy config.
        """
        input_names = output_names = None
        if deploy_cfg is not None:
            ir_config = get_ir_config(deploy_cfg)
            output_names = ir_config.get('output_names', None)
            input_names = ir_config.get('input_names', None)
        # TODO use input_names instead in the future for multiple inputs
        self.input_name = input_names[0] if input_names else 'input'
        self.output_names = output_names if output_names else ['output']
        super().__init__()

    @staticmethod
    def _build_wrapper(backend: Backend,
                       backend_files: Sequence[str],
                       device: str,
                       output_names: Optional[Sequence[str]] = None,
                       deploy_cfg: Optional[mmcv.Config] = None):
        """The default methods to build backend wrappers.

        Args:
            backend (Backend): The backend enum type.
            beckend_files (Sequence[str]): Paths to all required backend files(
                e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
            output_names (Sequence[str] | None): Names of model outputs in
                order. Defaults to `None` and the wrapper will load the output
                names from the model.
            deploy_cfg: Deployment config file.
        """
        if backend == Backend.ONNXRUNTIME:
            from mmdeploy.backend.onnxruntime import ORTWrapper
            return ORTWrapper(
                onnx_file=backend_files[0],
                device=device,
                output_names=output_names)
        elif backend == Backend.TENSORRT:
            from mmdeploy.backend.tensorrt import TRTWrapper
            return TRTWrapper(
                engine=backend_files[0], output_names=output_names)
        elif backend == Backend.PPLNN:
            from mmdeploy.backend.pplnn import PPLNNWrapper
            return PPLNNWrapper(
                onnx_file=backend_files[0],
                algo_file=backend_files[1],
                device=device,
                output_names=output_names)
        elif backend == Backend.NCNN:
            from mmdeploy.backend.ncnn import NCNNWrapper
            return NCNNWrapper(
                param_file=backend_files[0],
                bin_file=backend_files[1],
                output_names=output_names)
        elif backend == Backend.OPENVINO:
            from mmdeploy.backend.openvino import OpenVINOWrapper
            return OpenVINOWrapper(
                ir_model_file=backend_files[0], output_names=output_names)
        elif backend == Backend.SDK:
            assert deploy_cfg is not None, \
                'Building SDKWrapper requires deploy_cfg'
            from mmdeploy.backend.sdk import SDKWrapper
            task_name = SDK_TASK_MAP[get_task_type(deploy_cfg)]['cls_name']
            return SDKWrapper(
                model_file=backend_files[0],
                task_name=task_name,
                device=device)
        else:
            raise NotImplementedError(f'Unknown backend type: {backend.value}')

    @abstractmethod
    def forward(self, *args, **kwargs):
        """The forward interface that must be implemented.

        The arguments should align to forward() of the corresponding model of
        OpenMMLab codebases
        """
        pass

    @abstractmethod
    def show_result(self, *args, **kwargs):
        """The visualize interface that must be implemented.

        The arguments should align to show_result() of the corresponding model
        of OpenMMLab codebases
        """
        pass
