# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

from mmengine import Config
from mmengine.registry import Registry

from mmdeploy.utils import Codebase, Task, get_task_type
from .task import BaseTask


class MMCodebase(metaclass=ABCMeta):
    """Wrap the apis of OpenMMLab Codebase."""

    task_registry: Registry = None

    def __init__(self) -> None:
        pass

    @classmethod
    def get_task_class(cls, task: Task) -> BaseTask:
        """Get the task processors class according to the task type.

        Args:
            task (Task): The task enumeration.

        Returns:
            type: The task processor class.
        """
        return cls.task_registry.module_dict[task.value]

    @classmethod
    def build_task_processor(cls, model_cfg: Config, deploy_cfg: Config,
                             device: str):
        """The interface to build the task processors of the codebase.

        Args:
            model_cfg (str | Config): Model config file.
            deploy_cfg (str | Config): Deployment config file.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        task = get_task_type(deploy_cfg)
        return cls.task_registry.build(
            dict(
                type=task.value,
                model_cfg=model_cfg,
                deploy_cfg=deploy_cfg,
                device=device))

    @classmethod
    def register_deploy_modules(cls):
        """register deploy module."""
        raise NotImplementedError('register_deploy_modules not implemented.')

    @classmethod
    def register_all_modules(cls):
        """register codebase module."""
        raise NotImplementedError('register_all_modules not implemented.')


# Note that the build function returns the class instead of its instance.

CODEBASE = Registry('Codebases')


def get_codebase_class(codebase: Codebase) -> MMCodebase:
    """Get the codebase class from the registry.

    Args:
        codebase (Codebase): The codebase enum type.

    Returns:
        type: The codebase class
    """
    import importlib
    try:
        importlib.import_module(f'mmdeploy.codebase.{codebase.value}.deploy')
    except ImportError as e:
        from mmdeploy.utils import get_root_logger
        logger = get_root_logger()
        logger.warn(f'Import mmdeploy.codebase.{codebase.value}.deploy failed'
                    'Please check whether the module is the custom module.'
                    f'{e}')
    return CODEBASE.build({'type': codebase.value})
