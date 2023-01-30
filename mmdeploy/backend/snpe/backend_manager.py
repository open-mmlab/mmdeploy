# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import os.path as osp
from typing import Any, Optional, Sequence

from mmdeploy.utils import get_root_logger
from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('snpe')
class SNPEManager(BaseBackendManager):

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
        from .wrapper import SNPEWrapper
        uri = None
        if 'uri' in kwargs:
            uri = kwargs['uri']
        return SNPEWrapper(
            dlc_file=backend_files[0], uri=uri, output_names=output_names)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.

        Returns:
            bool: True if backend package is installed.
        """
        from .onnx2dlc import get_onnx2dlc_path
        onnx2dlc = get_onnx2dlc_path()
        if onnx2dlc is None:
            return False
        return osp.exists(onnx2dlc)

    @classmethod
    def to_backend(cls,
                   ir_files: Sequence[str],
                   work_dir: str,
                   log_level: int = logging.INFO,
                   device: str = 'cpu',
                   uri: str = '',
                   **kwargs) -> Sequence[str]:
        """Convert intermediate representation to given backend.

        Args:
            ir_files (Sequence[str]): The intermediate representation files.
            work_dir (str): The work directory, backend files and logs should
                be save in this directory.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            device (str, optional): The device type. Defaults to 'cpu'.

        Returns:
            Seqeuence[str]: Backend files.
        """
        from . import is_available
        logger = get_root_logger()

        if not is_available():
            logger.error('snpe support is not available, please check\n'
                         '1) `snpe-onnx-to-dlc` existed in `PATH`\n'
                         '2) snpe only support\n'
                         'ubuntu18.04')
            exit(1)

        from mmdeploy.apis.snpe import get_env_key, get_output_model_file
        from .onnx2dlc import from_onnx

        if get_env_key() not in os.environ:
            os.environ[get_env_key()] = uri

        backend_files = []
        for onnx_path in ir_files:
            dlc_path = get_output_model_file(onnx_path, work_dir)
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            from_onnx(onnx_path, osp.join(work_dir, onnx_name))
            backend_files += [dlc_path]

        return backend_files
