# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMDET3D_TASK = Registry('mmdet3d_tasks')


@CODEBASE.register_module(Codebase.MMDET3D.value)
class MMDetection3d(MMCodebase):
    """MMDetection3d codebase class."""

    task_registry = MMDET3D_TASK

    @classmethod
    def register_deploy_modules(cls):
        import mmdeploy.codebase.mmdet3d.models  # noqa: F401

    @classmethod
    def register_all_modules(cls):
        from mmdet3d.utils.setup_env import register_all_modules

        cls.register_deploy_modules()
        register_all_modules(True)
