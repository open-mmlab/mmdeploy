# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Optional, Sequence

from mmdeploy.utils import SDK_TASK_MAP, get_task_type
from ..base import BACKEND_MANAGERS, BaseBackendManager


@BACKEND_MANAGERS.register('sdk')
class SDKManager(BaseBackendManager):

    def build_wrapper(backend_files: Sequence[str],
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
        task_name = SDK_TASK_MAP[get_task_type(deploy_cfg)]['cls_name']
        return SDKWrapper(
            model_file=backend_files[0], task_name=task_name, device=device)
