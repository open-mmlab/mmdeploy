# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import sys
from typing import Any, Optional, Sequence

from mmdeploy.utils import get_backend_config, get_root_logger
from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('ncnn')
class NCNNManager(BaseBackendManager):

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
        from .wrapper import NCNNWrapper

        # For unittest deploy_config will not pass into _build_wrapper
        # function.
        if deploy_cfg:
            backend_config = get_backend_config(deploy_cfg)
            use_vulkan = backend_config.get('use_vulkan', False)
        else:
            use_vulkan = False
        return NCNNWrapper(
            param_file=backend_files[0],
            bin_file=backend_files[1],
            output_names=output_names,
            use_vulkan=use_vulkan)

    @classmethod
    def to_backend(cls,
                   ir_files: Sequence[str],
                   work_dir: str,
                   log_level: int = logging.INFO,
                   device: str = 'cpu',
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
        logger = get_root_logger()

        from . import is_available

        if not is_available():
            logger.error('ncnn support is not available, please make sure:\n'
                         '1) `mmdeploy_onnx2ncnn` existed in `PATH`\n'
                         '2) python import ncnn success')
            sys.exit(1)

        from mmdeploy.apis.ncnn import get_output_model_file
        from .onnx2ncnn import from_onnx

        backend_files = []
        for onnx_path in ir_files:
            model_param_path, model_bin_path = get_output_model_file(
                onnx_path, work_dir)
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            from_onnx(onnx_path, osp.join(work_dir, onnx_name))

            backend_files += [model_param_path, model_bin_path]

        return backend_files
