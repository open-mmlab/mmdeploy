# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMEDIT_TASK = Registry('mmedit_tasks')


@CODEBASE.register_module(Codebase.MMEDIT.value)
class MMEditing(MMCodebase):
    """mmediting codebase class."""

    task_registry = MMEDIT_TASK

    @classmethod
    def register_deploy_modules(cls):
        """register all rewriters for mmedit."""
        import mmdeploy.codebase.mmedit.models  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        """register all related modules and rewriters for mmedit."""
        from mmedit.utils.setup_env import register_all_modules

        cls.register_deploy_modules()
        register_all_modules(True)
