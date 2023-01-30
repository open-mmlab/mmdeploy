# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
from typing import Any, Optional, Sequence

from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('pplnn')
class PPLNNManager(BaseBackendManager):

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
        from .wrapper import PPLNNWrapper
        return PPLNNWrapper(
            onnx_file=backend_files[0],
            algo_file=backend_files[1] if len(backend_files) > 1 else None,
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
        ret = importlib.util.find_spec('pyppl') is not None

        return ret

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('pyppl').version
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
            Seqeuence[str]: Backend files.
        """
        from mmdeploy.utils import get_model_inputs
        from . import is_available
        from .onnx2pplnn import from_onnx
        assert is_available(), \
            'PPLNN is not available, please install PPLNN first.'

        pplnn_files = []
        for onnx_path in ir_files:
            algo_file = onnx_path.replace('.onnx', '.json')
            model_inputs = get_model_inputs(deploy_cfg)
            assert 'opt_shape' in model_inputs, 'Expect opt_shape ' \
                'in deploy config for PPLNN'
            # PPLNN accepts only 1 input shape for optimization,
            # may get changed in the future
            input_shapes = [model_inputs.opt_shape]
            algo_prefix = osp.splitext(algo_file)[0]
            from_onnx(
                onnx_path,
                algo_prefix,
                device=device,
                input_shapes=input_shapes)
            pplnn_files += [onnx_path, algo_file]

        return pplnn_files
