# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, get_task_type


def __build_mmseg_task(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                       device: str, registry: Registry) -> BaseTask:
    task = get_task_type(deploy_cfg)
    return registry.module_dict[task.value](model_cfg, deploy_cfg, device)


MMSEG_TASK = Registry('mmseg_tasks', build_func=__build_mmseg_task)


@CODEBASE.register_module(Codebase.MMSEG.value)
class MMSegmentation(MMCodebase):
    """mmsegmentation codebase class."""

    task_registry = MMSEG_TASK

    def __init__(self):
        super(MMSegmentation, self).__init__()

    @staticmethod
    def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                             device: str):
        """The interface to build the task processors of mmseg.

        Args:
            model_cfg (str | mmcv.Config): Model config file.
            deploy_cfg (str | mmcv.Config): Deployment config file.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        return MMSEG_TASK.build(model_cfg, deploy_cfg, device)

    @staticmethod
    def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                      dataset_type: str = 'val',
                      **kwargs) -> Dataset:
        """Build dataset for segmentation.

        Args:
            dataset_cfg (str | mmcv.Config): The input dataset config.
            dataset_type (str): A string represents dataset type, e.g.: 'train'
                , 'test', 'val'. Defaults to 'val'.

        Returns:
            Dataset: A PyTorch dataset.
        """
        from mmseg.datasets import build_dataset as build_dataset_mmseg

        assert dataset_type in dataset_cfg.data
        data_cfg = dataset_cfg.data[dataset_type]
        dataset = build_dataset_mmseg(data_cfg)
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
        """Build dataloader for segmentation.

        Args:
            dataset (Dataset): Input dataset.
            samples_per_gpu (int): Number of training samples on each GPU, i.e.
                ,batch size of each GPU.
            workers_per_gpu (int): How many subprocesses to use for data
                loading for each GPU.
            num_gpus (int): Number of GPUs. Only used in non-distributed
                training. dist (bool): Distributed training/test or not.
                Defaults  to `False`.
            dist (bool): Distributed training/test or not. Default: True.
            shuffle (bool): Whether to shuffle the data at every epoch.
                Defaults to `False`.
            seed (int): An integer set to be seed. Default is `None`.
            drop_last (bool): Whether to drop the last incomplete batch in
                epoch. Default to `False`.
            pin_memory (bool): Whether to use pin_memory in DataLoader.
                Default is `True`.
            persistent_workers (bool): If `True`, the data loader will not
                shutdown the worker processes after a dataset has been
                consumed once. This allows to maintain the workers Dataset
                instances alive. The argument also has effect in
                PyTorch>=1.7.0. Default is `True`.
            kwargs: Any other keyword argument to be used to initialize
                DataLoader.

        Returns:
            DataLoader: A PyTorch dataloader.
        """
        from mmseg.datasets import build_dataloader as build_dataloader_mmseg
        return build_dataloader_mmseg(
            dataset,
            samples_per_gpu,
            workers_per_gpu,
            num_gpus=num_gpus,
            dist=dist,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs)

    @staticmethod
    def single_gpu_test(model: torch.nn.Module,
                        data_loader: DataLoader,
                        show: bool = False,
                        out_dir: Optional[str] = None,
                        pre_eval: bool = True,
                        **kwargs):
        """Run test with single gpu.

        Args:
            model (torch.nn.Module): Input model from nn.Module.
            data_loader (DataLoader): PyTorch data loader.
            show (bool): Specifying whether to show plotted results. Defaults
                to `False`.
            out_dir (str): A directory to save results, defaults to `None`.
            pre_eval (bool): Use dataset.pre_eval() function to generate
                pre_results for metric evaluation. Mutually exclusive with
                efficient_test and format_results. Default: False.

        Returns:
            list: The prediction results.
        """
        from mmseg.apis import single_gpu_test
        outputs = single_gpu_test(
            model, data_loader, show, out_dir, pre_eval=pre_eval, **kwargs)
        return outputs
