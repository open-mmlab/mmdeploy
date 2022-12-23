# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from typing import Any, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('coreml')
class CoreMLManager(BaseBackendManager):

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
        from .wrapper import CoreMLWrapper
        return CoreMLWrapper(model_file=backend_files[0])

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
        from .torchscript2coreml import from_torchscript

        coreml_files = []
        for model_id, torchscript_path in enumerate(ir_files):
            torchscript_name = osp.splitext(osp.split(torchscript_path)[1])[0]
            output_file_prefix = osp.join(work_dir, torchscript_name)

            from_torchscript(model_id, torchscript_path, output_file_prefix,
                             deploy_cfg, coreml_files)

        return coreml_files
