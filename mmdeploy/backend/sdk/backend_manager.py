# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import sys
from typing import Any, Optional, Sequence

from mmdeploy.utils import get_file_path
from ..base import BACKEND_MANAGERS, BaseBackendManager

_is_available = False

module_name = 'mmdeploy_python'

candidates = [
    f'../../../build/lib/{module_name}.*.so',
    f'../../../build/bin/*/{module_name}.*.pyd'
]

lib_path = get_file_path(osp.dirname(__file__), candidates)

if lib_path:
    lib_dir = osp.dirname(lib_path)
    sys.path.append(lib_dir)

if importlib.util.find_spec(module_name) is not None:
    _is_available = True


@BACKEND_MANAGERS.register('sdk')
class SDKManager(BaseBackendManager):

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
        assert deploy_cfg is not None, \
            'Building SDKWrapper requires deploy_cfg'
        from mmdeploy.backend.sdk import SDKWrapper
        from mmdeploy.utils import SDK_TASK_MAP, get_task_type
        task_name = SDK_TASK_MAP[get_task_type(deploy_cfg)]['cls_name']
        return SDKWrapper(
            model_file=backend_files[0], task_name=task_name, device=device)

    @classmethod
    def is_available(cls, with_custom_ops: bool = False) -> bool:
        """Check whether backend is installed.

        Args:
            with_custom_ops (bool): check custom ops exists.

        Returns:
            bool: True if backend package is installed.
        """
        global _is_available

        return _is_available

    @classmethod
    def get_version(cls) -> str:
        """Get the version of the backend."""
        if not cls.is_available():
            return 'None'
        else:
            import pkg_resources
            try:
                return pkg_resources.get_distribution('mmdeploy').version
            except Exception:
                return 'None'
