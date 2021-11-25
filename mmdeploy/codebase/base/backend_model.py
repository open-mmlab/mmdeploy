from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence

import torch

from mmdeploy.utils.constants import Backend


class BaseBackendModel(torch.nn.Module, metaclass=ABCMeta):
    """A backend model wraps the details to initialize and run a backend
    engine."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _build_wrapper(backend: Backend,
                       backend_files: Sequence[str],
                       device: str,
                       output_names: Optional[Sequence[str]] = None):
        """The default methods to build backend wrappers.

        Args:
            backend (Backend): The backend enum type.
            beckend_files (Sequence[str]): Paths to all required backend files(
                e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
            output_names (Sequence[str] | None): Names of model outputs in
                order. Defaults to `None` and the wrapper will load the output
                names from the model.
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
        elif backend == Backend.PPL:
            from mmdeploy.backend.ppl import PPLWrapper
            return PPLWrapper(
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
