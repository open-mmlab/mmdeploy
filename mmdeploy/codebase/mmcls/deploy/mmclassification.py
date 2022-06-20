# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMCLS_TASK = Registry('mmcls_tasks')


@CODEBASE.register_module(Codebase.MMCLS.value)
class MMClassification(MMCodebase):
    """mmclassification codebase class."""

    task_registry = MMCLS_TASK
