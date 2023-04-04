# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from mmdeploy.ir.onnx import ONNXParam
from ..base import (BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam,
                    FileNameDescriptor)


@dataclass
class ONNXRuntimeParam(BaseBackendParam):
    """ONNX Runtime backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        device (str): The device used to perform the inference. Default to cpu.
    """
    file_name: FileNameDescriptor = FileNameDescriptor(
        default=None, postfix='.onnx')

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str), ('Expect string work_dir, '
                                                f'got {self.work_dir}')
        assert isinstance(self.file_name, str), ('Expect string file_name, '
                                                 f'got {self.file_name}')
        return osp.join(self.work_dir, self.file_name)


_BackendParam = ONNXRuntimeParam


@BACKEND_MANAGERS.register(
    'onnxruntime', param=_BackendParam, ir_param=ONNXParam)
class ONNXRuntimeManager(BaseBackendManager):

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.
        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        ret = importlib.util.find_spec('onnxruntime') is not None

        if ret and with_custom_ops:
            from .init_plugins import get_ops_path
            ops_path = get_ops_path()
            custom_ops_exist = osp.exists(ops_path)
            ret = ret and custom_ops_exist

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                ort_version = pkg_resources.get_distribution(
                    'onnxruntime').version
            except Exception:
                ort_version = 'None'
            try:
                ort_gpu_version = pkg_resources.get_distribution(
                    'onnxruntime-gpu').version
            except Exception:
                ort_gpu_version = 'None'

            if ort_gpu_version != 'None':
                return ort_gpu_version
            else:
                return ort_version

    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        """Check current environment.

        Returns:
            str: Info about the environment.
        """
        import pkg_resources

        try:
            if cls.is_available():
                ops_available = cls.is_available(with_custom_ops=True)
                ops_available = 'Available' \
                    if ops_available else 'NotAvailable'

                try:
                    ort_version = pkg_resources.get_distribution(
                        'onnxruntime').version
                except Exception:
                    ort_version = 'None'
                try:
                    ort_gpu_version = pkg_resources.get_distribution(
                        'onnxruntime-gpu').version
                except Exception:
                    ort_gpu_version = 'None'

                ort_info = f'ONNXRuntime:\t{ort_version}'
                log_callback(ort_info)
                ort_gpu_info = f'ONNXRuntime-gpu:\t{ort_gpu_version}'
                log_callback(ort_gpu_info)
                ort_ops_info = f'ONNXRuntime custom ops:\t{ops_available}'
                log_callback(ort_ops_info)

                info = f'{ort_info}\n{ort_gpu_info}\n{ort_ops_info}'
            else:
                info = 'ONNXRuntime:\tNone'
                log_callback(info)
        except Exception:
            info = f'{cls.backend_name}:\tCheckFailed'
            log_callback(info)
        return info

    @classmethod
    def to_backend(cls, onnx_path: str, save_path: str):
        """Convert intermediate representation to given backend.

        Args:
            onnx_path (str): The intermediate representation files.
            save_path (str): The save path of onnx path.
        Returns:
            Sequence[str]: Backend files.
        """
        if osp.abspath(save_path) != osp.abspath(onnx_path):
            shutil.copy(onnx_path, save_path)

    @classmethod
    def to_backend_from_param(cls, ir_model: str, param: BaseBackendParam):
        """Export to backend with packed backend parameter.

        Args:
            ir_model (str): The ir model path to perform the export.
            param (BaseBackendParam): Packed backend parameter.
        """
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        save_path = osp.join(param.work_dir, param.file_name)
        cls.to_backend(ir_model, save_path)

    @classmethod
    def build_wrapper(cls,
                      model_path: str,
                      device: str = 'cpu',
                      output_names: Optional[Sequence[str]] = None):
        """Build the wrapper for the backend model.

        Args:
            model_path (str): ONNX model file.
            device (str, optional): The device info. Defaults to 'cpu'.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
        """

        from .wrapper import ORTWrapper
        return ORTWrapper(
            onnx_file=model_path, device=device, output_names=output_names)

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (_BackendParam): Packed backend parameter.
        """
        assert isinstance(param, _BackendParam)
        assert isinstance(param.work_dir, str)
        assert isinstance(param.file_name, str)
        model_path = osp.join(param.work_dir, param.file_name)
        output_names = param.output_names
        if output_names is not None and len(output_names) == 0:
            output_names = None
        return cls.build_wrapper(
            model_path, device=param.device, output_names=output_names)

    @classmethod
    def build_param_from_config(cls,
                                config: Any,
                                work_dir: str,
                                backend_files: List[str] = None,
                                **kwargs) -> _BackendParam:
        """Build param from deploy config.

        Args:
            config (Any): The deploy config.
            work_dir (str): work directory of the parameters.
            backend_files (List[str]): The backend files of the model.

        Returns:
            BaseBackendParam: The packed backend parameter.
        """
        return _BackendParam(
            work_dir=work_dir, file_name=backend_files[0], **kwargs)
