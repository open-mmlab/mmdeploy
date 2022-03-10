# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import mmcv
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
        """The interface to build the task processors of mmdet3d.

        Args:
            model_cfg (str | mmcv.Config): Model config file.
            deploy_cfg (str | mmcv.Config): Deployment config file.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        return MMDET3D_TASK.build(model_cfg, deploy_cfg, device)

    @staticmethod
    def build_dataset(dataset_cfg: Union[str, mmcv.Config], *args,
                      **kwargs) -> Dataset:
        """Build dataset for detection3d.

        Args:
            dataset_cfg (str | mmcv.Config): The input dataset config.

        Returns:
            Dataset: A PyTorch dataset.
        """
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
                         runner_type: str = 'EpochBasedRunner',
                         persistent_workers: bool = True,
                         **kwargs) -> DataLoader:
        """Build dataloader for detection3d.

        Args:
            dataset (Dataset): Input dataset.
            samples_per_gpu (int): Number of training samples on each GPU, i.e.
                ,batch size of each GPU.
            workers_per_gpu (int): How many subprocesses to use for data
                loading for each GPU.
            num_gpus (int): Number of GPUs. Only used in non-distributed
                training.
            dist (bool): Distributed training/test or not.
                Defaults  to `False`.
            shuffle (bool): Whether to shuffle the data at every epoch.
                Defaults to `False`.
            seed (int): An integer set to be seed. Default is `None`.
            runner_type (str): Type of runner. Default: `EpochBasedRunner`.
            persistent_workers (bool): If True, the data loader will not
                shutdown the worker processes after a dataset has been consumed
                once. This allows to maintain the workers `Dataset` instances
                alive. This argument is only valid when PyTorch>=1.7.0.
                Default: False.
            kwargs: Any other keyword argument to be used to initialize
                DataLoader.

        Returns:
            DataLoader: A PyTorch dataloader.
        """
        from mmdet3d.datasets import \
            build_dataloader as build_dataloader_mmdet3d
        return build_dataloader_mmdet3d(
            dataset,
            samples_per_gpu,
            workers_per_gpu,
            num_gpus=num_gpus,
            dist=dist,
            shuffle=shuffle,
            seed=seed,
            runner_type=runner_type,
            persistent_workers=persistent_workers,
            **kwargs)
