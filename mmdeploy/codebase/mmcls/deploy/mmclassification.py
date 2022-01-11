# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, get_task_type


def __build_mmcls_task(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                       device: str, registry: Registry) -> BaseTask:
    task = get_task_type(deploy_cfg)
    return registry.module_dict[task.value](model_cfg, deploy_cfg, device)


MMCLS_TASK = Registry('mmcls_tasks', build_func=__build_mmcls_task)


@CODEBASE.register_module(Codebase.MMCLS.value)
class MMClassification(MMCodebase):
    """mmclassification codebase class."""

    task_registry = MMCLS_TASK

    def __init__(self):
        super(MMClassification, self).__init__()

    @staticmethod
    def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                             device: str) -> BaseTask:
        """The interface to build the task processors of mmseg.

        Args:
            model_cfg (mmcv.Config): Model config file.
            deploy_cfg (mmcv.Config): Deployment config file.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        return MMCLS_TASK.build(model_cfg, deploy_cfg, device)

    @staticmethod
    def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                      dataset_type: str = 'val',
                      **kwargs) -> Dataset:
        """Build dataset for classification.

        Args:
            dataset_cfg (str | mmcv.Config): The input dataset config.
            dataset_type (str): A string represents dataset type, e.g.: 'train'
                , 'test', 'val'.
                Default: 'val'.

        Returns:
            Dataset: A PyTorch dataset.
        """

        from mmcls.datasets import build_dataset as build_dataset_mmcls

        from mmdeploy.utils import load_config

        dataset_cfg = load_config(dataset_cfg)[0]
        data = dataset_cfg.data
        assert dataset_type in data

        dataset = build_dataset_mmcls(data[dataset_type])

        return dataset

    @staticmethod
    def build_dataloader(dataset: Dataset,
                         samples_per_gpu: int,
                         workers_per_gpu: int,
                         num_gpus: int = 1,
                         dist: bool = False,
                         shuffle: bool = False,
                         round_up: bool = True,
                         seed: Optional[int] = None,
                         pin_memory: bool = True,
                         persistent_workers: bool = True,
                         **kwargs) -> DataLoader:
        """Build dataloader for classifier.

        Args:
            dataset (Dataset): Input dataset.
            samples_per_gpu (int): Number of training samples on each GPU,
                i.e., batch size of each GPU.
            workers_per_gpu (int): How many subprocesses to use for data
                loading for each GPU.
            num_gpus (int): Number of GPUs. Only used in non-distributed
                training.
            dist (bool): Distributed training/test or not. Default: False.
            shuffle (bool): Whether to shuffle the data at every epoch.
                Default: False.
            round_up (bool): Whether to round up the length of dataset by
                adding extra samples to make it evenly divisible.
                Default: True.
            seed (int): An integer set to be seed. Default: None.
            pin_memory (bool): Whether to use pin_memory in DataLoader.
                Default: True.
            persistent_workers (bool): If `True`, the data loader will not
                shutdown the worker processes after a dataset has been
                consumed once. This allows to maintain the workers Dataset
                instances alive. The argument also has effect in
                PyTorch>=1.7.0. Default: True.
            kwargs: Any other keyword argument to be used to initialize
                DataLoader.

        Returns:
            DataLoader: A PyTorch dataloader.
        """
        from mmcls.datasets import build_dataloader as build_dataloader_mmcls
        return build_dataloader_mmcls(dataset, samples_per_gpu,
                                      workers_per_gpu, num_gpus, dist, shuffle,
                                      round_up, seed, pin_memory,
                                      persistent_workers, **kwargs)

    @staticmethod
    def single_gpu_test(model: torch.nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        win_name: str = '',
                        **kwargs) -> List:
        """Run test with single gpu.

        Args:
            model (torch.nn.Module): Input model from nn.Module.
            data_loader (DataLoader): PyTorch data loader.
            show (bool): Specifying whether to show plotted results.
                Default: False.
            out_dir (str): A directory to save results, Default: None.
            win_name (str): The name of windows, Default: ''.

        Returns:
            list: The prediction results.
        """
        from mmcls.apis import single_gpu_test
        outputs = single_gpu_test(
            model, data_loader, show, out_dir, win_name=win_name, **kwargs)
        return outputs
