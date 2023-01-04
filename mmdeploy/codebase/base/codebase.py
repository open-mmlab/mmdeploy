# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, Optional, Union

from mmengine import Config

from mmdeploy.utils import Codebase, Task, get_task_type
from .task import BaseTask, TaskRegistry


class BaseCodebase(metaclass=ABCMeta):
    """Wrap the apis of Codebase."""

    task_registry: TaskRegistry = None

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


MMCodebase = BaseCodebase


class CodebaseRegistry:
    """Codebase registry."""

    def __init__(self):
        self._module_dict = {}

    @property
    def module_dict(self):
        """get the module dict."""
        return self._module_dict

    def register_module(self, name: str, enum_name: Optional[str] = None):
        """register Codebase.

        Args:
            name (str): name of the codebase
            enum_name (Optional[str], optional): enum name of the codebase.
                if not given, the upper case of name would be used.
        """
        from mmdeploy.utils import get_root_logger
        logger = get_root_logger()

        if enum_name is None:
            enum_name = name.upper()

        def _wrap(cls):
            from mmdeploy.utils import Codebase

            if not hasattr(Codebase, enum_name):
                from aenum import extend_enum
                extend_enum(Codebase, enum_name, name)
                logger.info(f'Registry new codebase: {enum_name} = {name}.')

            if name in self._module_dict:
                logger.info(f'Codebase registry of `{name}`'
                            ' has already been registered.')

            self._module_dict[name] = cls

            return cls

        return _wrap

    def find(self, name: str) -> BaseCodebase:
        """Find the Codebase registry with name.

        Args:
            name (str): codebase name.
        Returns:
            BaseCodebase: Codebase registry of the given codebase.
        """
        import importlib

        # try import codebase if codebase is in `mmdeploy.codebase`
        try:
            importlib.import_module('mmdeploy.codebase.' + name)
        except Exception:
            pass
        return self._module_dict.get(name, None)

    def build(self, cfg: Dict, **kwargs):
        """build module.

        Args:
            cfg (Dict): The config of the module

        Returns:
            BaseCodebase: The output codebase instance.
        """
        assert 'type' in cfg, 'Can not get build, type not provided.'
        module_type = cfg['type']
        module_class = self.find(module_type)
        return module_class


CODEBASE = CodebaseRegistry()


def get_codebase_class(codebase: Union[str, Codebase]) -> MMCodebase:
    """Get the codebase class from the registry.

    Args:
        codebase (Codebase): The codebase enum type.

    Returns:
        type: The codebase class
    """
    import importlib

    if isinstance(codebase, Codebase):
        codebase = codebase.value

    try:
        importlib.import_module(f'mmdeploy.codebase.{codebase}.deploy')
    except ImportError as e:
        from mmdeploy.utils import get_root_logger
        logger = get_root_logger()
        logger.debug(f'Import mmdeploy.codebase.{codebase}.deploy failed.'
                     'Please check whether the module is the custom module.'
                     f'{e}')
    return CODEBASE.build({'type': codebase})
