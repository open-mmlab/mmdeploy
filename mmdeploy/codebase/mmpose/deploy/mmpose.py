# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, get_task_type, load_config


def __build_mmpose_task(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                        device: str, registry: Registry) -> BaseTask:
    task = get_task_type(deploy_cfg)
    return registry.module_dict[task.value](model_cfg, deploy_cfg, device)


MMPOSE_TASK = Registry('mmpose_tasks', build_func=__build_mmpose_task)


@CODEBASE.register_module(Codebase.MMPOSE.value, force=True)
class MMPose(MMCodebase):
    """mmpose codebase class."""

    task_registry = MMPOSE_TASK

    def __init__(self):
        super(MMCodebase, self).__init__()

    @staticmethod
    def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                             device: str) -> BaseTask:
        """The interface to build the task processors of mmpose.

        Args:
            model_cfg (mmcv.Config): Model config file.
            deploy_cfg (mmcv.Config): Deployment config file.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        return MMPOSE_TASK.build(model_cfg, deploy_cfg, device)

    @staticmethod
    def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                      dataset_type: str = 'test',
                      **kwargs) -> Dataset:
        """Build dataset for mmpose.

        Args:
            dataset_cfg (str | mmcv.Config): The input dataset config.
            dataset_type (str): A string represents dataset type, e.g.: 'train'
                , 'test', 'val'. Defaults to 'test'.

        Returns:
            Dataset: A PyTorch dataset.
        """
        from mmpose.datasets import build_dataset

        dataset_cfg = load_config(dataset_cfg)[0]
        assert dataset_type in dataset_cfg.data
        data_cfg = dataset_cfg.data[dataset_type]
        data_cfg.test_mode = True
        dataset = build_dataset(data_cfg, dict(test_mode=True))
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
                         **kwargs) -> DataLoader:
        """Build PyTorch DataLoader.

        Args:
            dataset (Dataset): A PyTorch dataset.
            samples_per_gpu (int): Number of training samples on each GPU,
                i.e., batch size of each GPU.
            workers_per_gpu (int): How many subprocesses to use for data
                loading for each GPU.
            num_gpus (int): Number of GPUs. Only used in non-distributed
                training.
            dist (bool): Distributed training/test or not. Default: True.
            shuffle (bool): Whether to shuffle the data at every epoch.
                Default: False.
            seed (int): An integer set to be seed. Default is ``None``.
            drop_last (bool): Whether to drop the last incomplete batch
                in epoch. Default: False.
            pin_memory (bool): Whether to use pin_memory in DataLoader.
                Default: True.
            kwargs: Other keyword arguments to be used to initialize
                DataLoader.

        Returns:
            DataLoader: A PyTorch dataloader.
        """
        from mmpose.datasets import build_dataloader
        return build_dataloader(
            dataset,
            samples_per_gpu,
            workers_per_gpu,
            num_gpus=num_gpus,
            dist=dist,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            pin_memory=pin_memory,
            **kwargs)

    @staticmethod
    def single_gpu_test(model: torch.nn.Module, data_loader: DataLoader,
                        show: bool, out_dir: str, **kwargs) -> list:
        """Run test with single gpu.

        Args:
            model (torch.nn.Module): Input model from nn.Module.
            data_loader (DataLoader): PyTorch data loader.
            show (bool): Specifying whether to show plotted results. Defaults
                to ``False``.
            out_dir (str): A directory to save results, defaults to ``None``.
        Returns:
            list: The prediction results.
        """
        from mmpose.apis import single_gpu_test
        return single_gpu_test(model, data_loader)
