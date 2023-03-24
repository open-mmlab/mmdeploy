# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, get_task_type, load_config


def __build_mmflow_task(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                        device: str, registry: Registry) -> BaseTask:
    task = get_task_type(deploy_cfg)
    return registry.module_dict[task.value](model_cfg, deploy_cfg, device)


MMFLOW_TASK = Registry('mmflow_tasks', build_func=__build_mmflow_task)


@CODEBASE.register_module(Codebase.MMFLOW.value)
class MMFlow(MMCodebase):
    """mmediting codebase class."""

    task_registry = MMFLOW_TASK

    def __init__(self):
        super().__init__()

    @staticmethod
    def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                             device: str) -> BaseTask:
        """The interface to build the task processors of mmedit.

        Args:
            model_cfg (mmcv.Config): Model config file.
            deploy_cfg (mmcv.Config): Deployment config file.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        return MMFLOW_TASK.build(model_cfg, deploy_cfg, device)

    @staticmethod
    def build_dataset(dataset_cfg: Union[str, mmcv.Config], *args,
                      **kwargs) -> Dataset:
        """Build dataset for processor.

        Args:
            dataset_cfg (str | mmcv.Config): The input dataset config.

        Returns:
            Dataset: A PyTorch dataset.
        """
        from mmflow.datasets import build_dataset as build_dataset_mmflow

        dataset_cfg = load_config(dataset_cfg)[0]
        data = dataset_cfg.data

        dataset = build_dataset_mmflow(data.test)
        return dataset

    @staticmethod
    def build_dataloader(dataset: Dataset,
                         samples_per_gpu: int,
                         workers_per_gpu: int,
                         sample_ratio: Optional[Sequence] = None,
                         num_gpus: int = 1,
                         dist: bool = False,
                         shuffle: bool = False,
                         seed: Optional[int] = None,
                         persistent_workers: bool = True,
                         **kwargs) -> DataLoader:
        """Build PyTorch DataLoader.

        In distributed training, each GPU/process has a dataloader.
        In non-distributed training, there is only one dataloader for all GPUs.

        Args:
            dataset (:obj:`Dataset`): A PyTorch dataset.
            samples_per_gpu (int): Number of samples on each GPU, i.e.,
                batch size of each GPU.
            workers_per_gpu (int): How many subprocesses to use for data
                loading for each GPU.
            num_gpus (int): Number of GPUs. Only used in non-distributed
                training. Default: 1.
            dist (bool): Distributed training/test or not. Default: True.
            shuffle (bool): Whether to shuffle the data at every epoch.
                Default: True.
            seed (int | None): Seed to be used. Default: None.
            drop_last (bool): Whether to drop the last incomplete batch
                in epoch.　Default: False.
            pin_memory (bool): Whether to use pin_memory in DataLoader.
                Default: True.
            persistent_workers (bool): If True, the data loader will not
                shutdown　the worker processes after a dataset has been
                consumed once.
                This allows to maintain the workers Dataset instances alive.
                The argument also has effect in PyTorch>=1.7.0.
                Default: True.
            kwargs (dict, optional): Any keyword argument to be used to
                initialize　DataLoader.

        Returns:
            DataLoader: A PyTorch dataloader.
        """
        from mmflow.datasets import build_dataloader as build_dataloader_mmflow
        return build_dataloader_mmflow(dataset, samples_per_gpu,
                                       workers_per_gpu, sample_ratio, num_gpus,
                                       dist, shuffle, seed, persistent_workers,
                                       **kwargs)

    @staticmethod
    def single_gpu_test(model: torch.nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        **kwargs) -> list:
        """Run test with single gpu.

        Args:
            model (torch.nn.Module): Input model from nn.Module.
            data_loader (DataLoader): PyTorch data loader.
            show (bool): Specifying whether to show plotted results.
                Default: False.
            out_dir (str): A directory to save results, Default: None.

        Returns:
            list: The prediction results.
        """
        from mmflow.apis import single_gpu_test
        show_dir = out_dir
        outputs = single_gpu_test(model, data_loader, out_dir, show_dir)
        return outputs
