# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.utils import Codebase

MMOCR_TASK = Registry('mmocr_tasks')


@CODEBASE.register_module(Codebase.MMOCR.value)
class MMOCR(MMCodebase):
    """MMOCR codebase class."""

    task_registry = MMOCR_TASK

    @classmethod
    def register_all_modules(cls):
        from mmdet.utils.setup_env import \
            register_all_modules as register_all_modules_mmdet
        from mmocr.utils.setup_env import \
            register_all_modules as register_all_modules_mmocr
        register_all_modules_mmocr(True)
        register_all_modules_mmdet(False)
