# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Optional, Sequence

from mmdeploy.utils import get_backend_config
from ..base import BACKEND_MANAGERS, BaseBackendManager


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
            popef_file=backend_files[0],
            bps=bps,
            output_names=output_names
        )
