# Copyright (c) OpenMMLab. All rights reserved.

import mmengine
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, get_task_type


def __build_mmaction_task(model_cfg: mmengine.Config,
                          deploy_cfg: mmengine.Config, device: str,
                          registry: Registry) -> BaseTask:
    task = get_task_type(deploy_cfg)
    return registry.module_dict[task.value](model_cfg, deploy_cfg, device)


MMACTION_TASK = Registry('mmaction_tasks', build_func=__build_mmaction_task)


@CODEBASE.register_module(Codebase.MMACTION.value)
class MMACTION(MMCodebase):
    """MMAction codebase class."""

    task_registry = MMACTION_TASK

    @staticmethod
    def build_task_processor(model_cfg: mmengine.Config,
                             deploy_cfg: mmengine.Config, device: str):
        """The interface to build the task processors of mmaction.

        Args:
            model_cfg (str | mmengine.Config): Model config file.
            deploy_cfg (str | mmengine.Config): Deployment config file.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        return MMACTION_TASK.build(model_cfg, deploy_cfg, device)
