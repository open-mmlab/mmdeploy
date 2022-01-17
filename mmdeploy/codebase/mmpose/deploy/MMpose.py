from typing import Optional, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, get_task_type


def __build_mmpose_task(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                        device: str, registry: Registry) -> BaseTask:
    task = get_task_type(deploy_cfg)
    return registry.module_dict[task.value](model_cfg, deploy_cfg, device)


MMPOSE_TASK = Registry('mmpose_tasks', build_func=__build_mmpose_task)


@CODEBASE.register_module(Codebase.MMPOSE.value, force=True)
class MMPose(MMCodebase):

    task_registry = MMPOSE_TASK

    def __init__(self):
        super().__init__()

    @staticmethod
    def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                             device: str) -> BaseTask:
        return MMPOSE_TASK.build(model_cfg, deploy_cfg, device)

    @staticmethod
    def single_gpu_test(model: torch.nn.Module,
                        data_loader: DataLoader,
                        save_image: bool = False,
                        save_path: Optional[str] = None,
                        iteration: int = None) -> list:

        from mmpose.apis import single_gpu_test
        return single_gpu_test(model, data_loader)
