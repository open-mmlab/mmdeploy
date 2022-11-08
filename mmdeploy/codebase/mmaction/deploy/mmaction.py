# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMACTION_TASK = Registry('mmaction_tasks')


@CODEBASE.register_module(Codebase.MMACTION.value)
class MMACTION(MMCodebase):
    """MMAction codebase class."""

    task_registry = MMACTION_TASK

    @classmethod
    def register_deploy_modules(cls):
        import mmdeploy.codebase.mmaction.models  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        from mmaction.utils.setup_env import register_all_modules
        cls.register_deploy_modules()
        register_all_modules(True)
