# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from typing import Any, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('ascend')
class AscendManager(BaseBackendManager):

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
        from .wrapper import AscendWrapper
        return AscendWrapper(model=backend_files[0], device=device)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.

        Returns:
            bool: True if backend package is installed.
        """
        import importlib
        return importlib.util.find_spec('acl') is not None

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('acl').version
            except Exception:
                return 'None'

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
            Sequence[str]: Backend files.
        """
        from mmdeploy.utils import get_model_inputs
        from .onnx2ascend import from_onnx

        model_inputs = get_model_inputs(deploy_cfg)

        om_files = []
        for model_id, onnx_path in enumerate(ir_files):
            om_name = osp.splitext(osp.split(onnx_path)[1])[0] + '.om'
            om_path = osp.join(work_dir, om_name)
            from_onnx(onnx_path, work_dir, model_inputs[model_id])
            om_files.append(om_path)
        backend_files = om_files

        return backend_files
