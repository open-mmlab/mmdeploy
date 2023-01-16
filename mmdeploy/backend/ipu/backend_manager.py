# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from typing import Any, Optional, Sequence, Callable

from mmdeploy.utils import get_backend_config
from ..base import BACKEND_MANAGERS, BaseBackendManager
from .converter import onnx_to_popef
import sys


@BACKEND_MANAGERS.register('ipu')
class IPUManager(BaseBackendManager):

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
        from .wrapper import IPUWrapper

        # For unittest deploy_config will not pass into _build_wrapper
        # function.
        if deploy_cfg:
            backend_config = get_backend_config(deploy_cfg)
            bps = backend_config.get('batches_per_step', 1)
        else:
            bps = 1
        return IPUWrapper(
            popef_file=backend_files[0], bps=bps, output_names=output_names)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.

        Returns:
            bool: True if backend package is installed.
        """
        try:
            if 'onnx' in sys.modules.keys():
                del sys.modules['onnx']
                import popart
            else:
                import popart

            deviceManager = popart.DeviceManager()
            deviceManager.acquireAvailableDevice(1)
            return True
        except Exception as e:
            print('IPU environment is not set', str(e))
            return False

    @classmethod
    def check_env(cls, log_callback: Callable = lambda _: _) -> str:
        """Check current environment.

        Returns:
            str: Info about the environment.
        """
        available = cls.is_available()
        available_info = 'Available' if available else 'NotAvailable'
        info = f'IPU:\t{available_info}'
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

        backend_files = []
        for model_id, onnx_path in enumerate(ir_files):
            model_name = onnx_path.split('/')[-1][:-5]
            ipu_config = deploy_cfg.get('backend_config', {})
            output_dir = ipu_config.get('output_dir', '')
            # assert output_dir != '', 'output dir for ipu backend is not set'
            # assert os.path.exists(output_dir), 'output dir not exist'

            if output_dir == '':
                output_dir = work_dir

            model_dir = os.path.join(output_dir, model_name)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            ipu_config['output_dir'] = model_dir
            onnx_to_popef(onnx_path, ipu_config)
            backend_files.append(os.path.join(model_dir, 'executable.popef'))

        return backend_files
