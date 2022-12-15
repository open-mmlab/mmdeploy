# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMACTION2_TASK = Registry('mmaction2_tasks')


@CODEBASE.register_module(Codebase.MMACTION2.value)
class MMACTION2(MMCodebase):
    """MMAction2 codebase class."""

    task_registry = MMACTION2_TASK

    @classmethod
    def register_deploy_modules(cls):
        import mmdeploy.codebase.mmaction2.models  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        from mmaction.utils.setup_env import register_all_modules
        cls.register_deploy_modules()
        register_all_modules(True)
