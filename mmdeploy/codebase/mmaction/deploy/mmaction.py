# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMACTION_TASK = Registry('mmaction_tasks')


@CODEBASE.register_module(Codebase.MMACTION.value)
class MMACTION(MMCodebase):
    """MMAction codebase class."""

    task_registry = MMACTION_TASK
