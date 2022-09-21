# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.registry import Registry

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.codebase.mmdet.deploy import ObjectDetection
from mmdeploy.utils import Codebase, Task

MMYOLO_TASK = Registry('mmyolo_tasks')


@CODEBASE.register_module(Codebase.MMYOLO.value)
class MMYolo(MMCodebase):
    """MMDetection codebase class."""

    task_registry = MMYOLO_TASK


@MMYOLO_TASK.register_module(Task.OBJECT_DETECTION.value)
class YoloObjectDetection(ObjectDetection):

    def get_visualizer(self, name: str, save_dir: str):
        from mmdet.visualization import DetLocalVisualizer  # noqa: F401,F403
        visualizer = super().get_visualizer(name, save_dir)
        return visualizer
