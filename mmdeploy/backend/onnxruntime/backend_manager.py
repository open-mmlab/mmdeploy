# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from typing import Any, Callable, Optional, Sequence

from mmdeploy.utils import get_backend_config, get_common_config
from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('onnxruntime')
class ONNXRuntimeManager(BaseBackendManager):

    @classmethod
    def build_wrapper(cls,
                      backend_files: Sequence[str],
                      device: str = 'cpu',
                      input_names: Optional[Sequence[str]] = None,
                      output_names: Optional[Sequence[str]] = None,
                      deploy_cfg: Optional[Any] = None,
                      **kwargs):
        """Build the wrapper for the backend model.

        Args:
            backend_files (Sequence[str]): Backend files.
            device (str, optional): The device info. Defaults to 'cpu'.
            input_names (Optional[Sequence[str]], optional): input names.
                Defaults to None.
            output_names (Optional[Sequence[str]], optional): output names.
                Defaults to None.
            deploy_cfg (Optional[Any], optional): The deploy config. Defaults
                to None.
        """

        from .wrapper import ORTWrapper
        return ORTWrapper(
            onnx_file=backend_files[0],
            device=device,
            output_names=output_names)

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
    def to_backend(cls,
                   ir_files: Sequence[str],
                   work_dir: str,
                   deploy_cfg: Any,
                   log_level: int = logging.INFO,
                   device: str = 'cpu',
                   **kwargs) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            ir_files (Sequence[str]): The intermediate representation files.
            work_dir (str): The work directory, backend files and logs should
                be saved in this directory.
            deploy_cfg (Any): The deploy config.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            device (str, optional): The device type. Defaults to 'cpu'.
        Returns:
            Sequence[str]: Backend files.
        """
        backend_cfg = get_backend_config(deploy_cfg)

        precision = backend_cfg.get('precision', 'fp32')
        if precision == 'fp16':
            import onnx
            from onnxconverter_common import float16

            common_cfg = get_common_config(deploy_cfg)
            model = onnx.load(ir_files[0])
            model_fp16 = float16.convert_float_to_float16(model, **common_cfg)
            onnx.save(model_fp16, ir_files[0])
        return ir_files
