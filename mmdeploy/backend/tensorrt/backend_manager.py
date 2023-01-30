# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from typing import Any, Callable, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('tensorrt')
class TensorRTManager(BaseBackendManager):

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

        from .wrapper import TRTWrapper
        return TRTWrapper(engine=backend_files[0], output_names=output_names)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.

        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        ret = importlib.util.find_spec('tensorrt') is not None

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
                return pkg_resources.get_distribution('tensorrt').version
            except Exception:
                return 'None'

    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        """Check current environment.

        Returns:
            str: Info about the environment.
        """
        info = super().check_env(log_callback=log_callback)
        available = cls.is_available()
        ops_available = cls.is_available(with_custom_ops=True)
        ops_available = 'Available' if ops_available else 'NotAvailable'

        if available:
            ops_info = f'tensorrt custom ops:\t{ops_available}'
            log_callback(ops_info)
            info = f'{info}\n{ops_info}'
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
                be save in this directory.
            deploy_cfg (Any): The deploy config.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            device (str, optional): The device type. Defaults to 'cpu'.

        Returns:
            Seqeuence[str]: Backend files.
        """
        import os.path as osp

        from mmdeploy.utils import get_model_inputs, get_partition_config
        model_params = get_model_inputs(deploy_cfg)
        partition_cfgs = get_partition_config(deploy_cfg)
        assert len(model_params) == len(ir_files)

        from . import is_available
        assert is_available(), (
            'TensorRT is not available,'
            ' please install TensorRT and build TensorRT custom ops first.')

        from .onnx2tensorrt import onnx2tensorrt
        backend_files = []
        for model_id, model_param, onnx_path in zip(
                range(len(ir_files)), model_params, ir_files):
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            save_file = model_param.get('save_file', onnx_name + '.engine')

            partition_type = 'end2end' if partition_cfgs is None \
                else onnx_name
            onnx2tensorrt(
                work_dir,
                save_file,
                model_id,
                deploy_cfg,
                onnx_path,
                device=device,
                partition_type=partition_type)

            backend_files.append(osp.join(work_dir, save_file))

        return backend_files
