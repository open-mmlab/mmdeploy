# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMAGIC_TASK = Registry('mmagic_tasks')


@CODEBASE.register_module(Codebase.MMAGIC.value)
class MMagic(MMCodebase):
    """mmagic codebase class."""

    task_registry = MMAGIC_TASK

    @classmethod
    def register_deploy_modules(cls):
        """register all rewriters for mmagic."""
        import mmdeploy.codebase.mmagic.models  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        """register all related modules and rewriters for mmagic."""
        from mmagic.utils.setup_env import register_all_modules

        cls.register_deploy_modules()
        register_all_modules(True)
