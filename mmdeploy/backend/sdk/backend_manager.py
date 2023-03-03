# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import sys
from dataclasses import dataclass
from typing import Any, List

from mmdeploy.utils import get_file_path
from ..base import BACKEND_MANAGERS, BaseBackendManager, BaseBackendParam

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


@dataclass
class SDKParam(BaseBackendParam):
    """SDK backend parameters.

    Args:
        work_dir (str): The working directory.
        file_name (str): File name of the serialized model. Postfix will be
            added automatically.
        task_name (str): The name of the SDK task.
        device (str): Inference device.
    """
    _default_postfix = ''

    task_name: str = None

    def get_model_files(self) -> str:
        """get the model files."""
        assert isinstance(self.work_dir, str)
        assert isinstance(self.file_name, str)
        model_path = osp.join(self.work_dir, self.file_name)
        return model_path


_BackendParam = SDKParam


@BACKEND_MANAGERS.register('sdk', param=SDKParam)
class SDKManager(BaseBackendManager):

    @classmethod
    def build_wrapper(cls,
                      backend_model: str,
                      task_name: str,
                      device: str = 'cpu'):
        """Build the wrapper for the backend model.

        Args:
            backend_model (str): Backend model.
            task_name (str): The name of the SDK task.
            device (str, optional): The device info. Defaults to 'cpu'.
            deploy_cfg (Optional[Any], optional): The deploy config. Defaults
                to None.
        """
        from .wrapper import SDKWrapper
        return SDKWrapper(
            model_file=backend_model, task_name=task_name, device=device)

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

    @classmethod
    def build_wrapper_from_param(cls, param: _BackendParam):
        """Export to backend with packed backend parameter.

        Args:
            param (BaseBackendParam): Packed backend parameter.
        """
        model_path = param.get_model_files()
        device = param.device
        task_name = param.task_name
        return cls.build_wrapper(
            model_path, task_name=task_name, device=device)

    @classmethod
    def build_param_from_config(cls,
                                config: Any,
                                work_dir: str,
                                backend_files: List[str] = None,
                                **kwargs) -> _BackendParam:
        from mmdeploy.utils import SDK_TASK_MAP, get_task_type
        task_name = SDK_TASK_MAP[get_task_type(config)]['cls_name']

        kwargs.setdefault('task_name', task_name)
        return _BackendParam(
            work_dir=work_dir, file_name=backend_files[0], **kwargs)
