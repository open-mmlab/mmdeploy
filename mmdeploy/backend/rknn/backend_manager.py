# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from typing import Any, Optional, Sequence

from mmdeploy.utils import get_common_config
from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('rknn')
class RKNNManager(BaseBackendManager):

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

        from .wrapper import RKNNWrapper
        common_config = get_common_config(deploy_cfg)
        return RKNNWrapper(
            model=backend_files[0],
            common_config=common_config,
            input_names=input_names,
            output_names=output_names)

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
        from . import is_available
        assert is_available(
        ), 'RKNN is not available, please install RKNN first.'

        from .onnx2rknn import onnx2rknn

        backend_files = []
        for model_id, onnx_path in zip(range(len(ir_files)), ir_files):
            pre_fix_name = osp.splitext(osp.split(onnx_path)[1])[0]
            output_path = osp.join(work_dir, pre_fix_name + '.rknn')
            onnx2rknn(onnx_path, output_path, deploy_cfg)
            backend_files.append(output_path)

        return backend_files
