# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.device import get_device
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.runner import Runner


class DeployTestRunner(Runner):
    """The runner for test models.

    Args:
        log_file (str | None): The path of log file. Default is ``None``.
        device (str): The device type.
    """

    def __init__(self,
                 log_file: Optional[str] = None,
                 device: str = get_device(),
                 *args,
                 **kwargs):

        self._log_file = log_file
        self._device = device
        super(DeployTestRunner, self).__init__(*args, **kwargs)

    def wrap_model(self, model_wrapper_cfg: Optional[Dict],
                   model: BaseModel) -> BaseModel:
        """Wrap the model to :obj:``MMDistributedDataParallel`` or other custom
        distributed data-parallel module wrappers.

        An example of ``model_wrapper_cfg``::

            model_wrapper_cfg = dict(
                broadcast_buffers=False,
                find_unused_parameters=False
            )

        Args:
            model_wrapper_cfg (dict, optional): Config to wrap model. If not
                specified, ``DistributedDataParallel`` will be used in
                distributed environment. Defaults to None.
            model (BaseModel): Model to be wrapped.

        Returns:
            BaseModel or DistributedDataParallel: BaseModel or subclass of
            ``DistributedDataParallel``.
        """
        return model.to(self._device)

    def build_logger(self,
                     log_level: Union[int, str] = 'INFO',
                     log_file: str = None,
                     **kwargs) -> MMLogger:
        """Build a global accessible MMLogger.

        Args:
            log_level (int or str): The log level of MMLogger handlers.
                Defaults to 'INFO'.
            log_file (str, optional): Path of filename to save log.
                Defaults to None.
            **kwargs: Remaining parameters passed to ``MMLogger``.

        Returns:
            MMLogger: A MMLogger object build from ``logger``.
        """
        if log_file is None:
            log_file = self._log_file

        return super().build_logger(log_level, log_file, **kwargs)
