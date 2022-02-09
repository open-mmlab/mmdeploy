# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import mmcv
import torch
from mmcv.utils.registry import Registry
from torch.utils.data import DataLoader, Dataset

from mmdeploy.utils import Codebase, Task
from .task import BaseTask


class MMCodebase(metaclass=ABCMeta):
    """Wrap the apis of OpenMMLab Codebase."""

    task_registry: Registry = None

    def __init__(self) -> None:
        pass

    @classmethod
    def get_task_class(cls, task: Task) -> BaseTask:
        """Get the task processors class according to the task type.

        Args:
            task (Task): The task enumeration.

        Returns:
            type: The task processor class.
        """
        return cls.task_registry.module_dict[task.value]

    @staticmethod
    @abstractmethod
    def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                             device: str):
        """The interface to build the task processors of the codebase.

        Args:
            model_cfg (str | mmcv.Config): Model config file.
            deploy_cfg (str | mmcv.Config): Deployment config file.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        pass

    @staticmethod
    @abstractmethod
    def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                      dataset_type: str = 'val',
                      **kwargs) -> Dataset:
        """Build dataset for different codebase.

        Args:
            dataset_cfg (str | mmcv.Config): Dataset config file or Config
                object.
            dataset_type (str): Specifying dataset type, e.g.: 'train', 'test',
                'val', defaults to 'val'.

        Returns:
            Dataset: The built dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def build_dataloader(dataset: Dataset, samples_per_gpu: int,
                         workers_per_gpu: int, **kwargs) -> DataLoader:
        """Build PyTorch dataloader.

        Args:
            dataset (Dataset): A PyTorch dataset.
            samples_per_gpu (int): Number of training samples on each GPU,
                i.e., batch size of each GPU.
            workers_per_gpu (int): How many subprocesses to use for data
                loading for each GPU.

        Returns:
            DataLoader: A PyTorch dataloader.
        """
        pass

    @staticmethod
    @abstractmethod
    def single_gpu_test(model: torch.nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        **kwargs):
        """Run test with single gpu.

        Args:
            model (torch.nn.Module): Input model from nn.Module.
            data_loader (DataLoader): PyTorch data loader.
            show (bool): Specifying whether to show plotted results. Defaults
                to `False`.
            out_dir (str): A directory to save results, defaults to `None`.

        Returns:
            list: The prediction results.
        """
        pass


# Note that the build function returns the class instead of its instance.


def __build_codebase_class(codebase: Codebase, registry: Registry):
    return registry.module_dict[codebase.value]


CODEBASE = Registry('Codebases', build_func=__build_codebase_class)


def get_codebase_class(codebase: Codebase) -> MMCodebase:
    """Get the codebase class from the registry.

    Args:
        codebase (Codebase): The codebase enum type.

    Returns:
        type: The codebase class
    """
    return CODEBASE.build(codebase)
