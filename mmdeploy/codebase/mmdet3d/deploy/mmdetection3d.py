# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, get_task_type


def __build_mmdet3d_task(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                         device: str, registry: Registry) -> BaseTask:
    task = get_task_type(deploy_cfg)
    return registry.module_dict[task.value](model_cfg, deploy_cfg, device)


MMDET3D_TASK = Registry('mmdet3d_tasks', build_func=__build_mmdet3d_task)


@CODEBASE.register_module(Codebase.MMDET3D.value)
class MMDetection3d(MMCodebase):
    task_registry = MMDET3D_TASK

    def __init__(self):
        super().__init__()

    @staticmethod
    def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                             device: str) -> BaseTask:
        return MMDET3D_TASK.build(model_cfg, deploy_cfg, device)

    @staticmethod
    def build_dataset(dataset_cfg: Union[str, mmcv.Config], *args,
                      **kwargs) -> Dataset:

        from mmdet3d.datasets import build_dataset as build_dataset_mmdet3d
        from mmdeploy.utils import load_config
        dataset_cfg = load_config(dataset_cfg)[0]
        data = dataset_cfg.data

        dataset = build_dataset_mmdet3d(data.test)
        return dataset

    @staticmethod
    def build_dataloader(dataset: Dataset,
                         samples_per_gpu: int,
                         workers_per_gpu: int,
                         num_gpus: int = 1,
                         dist: bool = False,
                         shuffle: bool = False,
                         seed: Optional[int] = None,
                         drop_last: bool = False,
                         pin_memory: bool = True,
                         persistent_workers: bool = True,
                         **kwargs) -> DataLoader:
        from mmdet3d.datasets import build_dataloader as build_dataloader_mmdet3d
        return build_dataloader_mmdet3d(
            dataset,
            samples_per_gpu,
            workers_per_gpu,
            num_gpus=num_gpus,
            dist=dist,
            shuffle=shuffle,
            seed=seed,
            **kwargs)

    @staticmethod
    def single_gpu_test(model: torch.nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        **kwargs):
        from mmdet3d.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir, kwargs)
        return outputs


# if __name__ == '__main__':
#     model_cfg = '../workspace/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
#     checkpoint = '../workspace/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth'
#     codebase = MMDetection3d()
#     dataset = codebase.build_dataset(model_cfg)
#     dataloader = codebase.build_dataloader(dataset,1,1)
#     from mmdet3d.apis import init_model
#     model = init_model(model_cfg,checkpoint)
#     print(codebase.single_gpu_test(model,dataloader))
