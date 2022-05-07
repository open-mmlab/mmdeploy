# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import mmcv
import torch
from mmcv.utils import Registry
from mmdet.datasets import replace_ImageToTensor
from torch.utils.data import DataLoader, Dataset

from mmdeploy.codebase.base import CODEBASE, BaseTask, MMCodebase
from mmdeploy.utils import Codebase, get_task_type


def __build_mmrotate_task(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                          device: str, registry: Registry) -> BaseTask:
    task = get_task_type(deploy_cfg)
    return registry.module_dict[task.value](model_cfg, deploy_cfg, device)


MMROTATE_TASK = Registry('mmrotate_tasks', build_func=__build_mmrotate_task)


@CODEBASE.register_module(Codebase.MMROTATE.value)
class MMROTATE(MMCodebase):
    """mmrotate codebase class."""

    task_registry = MMROTATE_TASK

    def __init__(self):
        super(MMROTATE, self).__init__()

    @staticmethod
    def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                             device: str):
        """The interface to build the task processors of mmrotate.

        Args:
            model_cfg (str | mmcv.Config): Model config file or loaded Config
                object.
            deploy_cfg (str | mmcv.Config): Deployment config file or loaded
                Config object.
            device (str): A string specifying device type.

        Returns:
            BaseTask: A task processor.
        """
        return MMROTATE_TASK.build(model_cfg, deploy_cfg, device)

    @staticmethod
    def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                      dataset_type: str = 'val',
                      **kwargs) -> Dataset:
        """Build dataset for mmrotate.

        Args:
            dataset_cfg (str | mmcv.Config): The input dataset config.
            dataset_type (str): A string represents dataset type, e.g.: 'train'
                , 'test', 'val'. Defaults to 'val'.

        Returns:
            Dataset: A PyTorch dataset.
        """
        from mmrotate.datasets import build_dataset as build_dataset_mmrotate

        # dataset_cfg = load_config(dataset_cfg)[0]
        assert dataset_type in dataset_cfg.data
        data_cfg = dataset_cfg.data[dataset_type]
        # in case the dataset is concatenated
        if isinstance(data_cfg, dict):
            data_cfg.test_mode = True
            samples_per_gpu = data_cfg.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                data_cfg.pipeline = replace_ImageToTensor(data_cfg.pipeline)
        elif isinstance(data_cfg, list):
            for ds_cfg in data_cfg:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in data_cfg])
            if samples_per_gpu > 1:
                for ds_cfg in data_cfg:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
        dataset = build_dataset_mmrotate(data_cfg)
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
                         persistent_workers: bool = True,
                         **kwargs) -> DataLoader:
        """Build dataloader for mmrotate.

        Args:
            dataset (Dataset): Input dataset.
            samples_per_gpu (int): Number of training samples on each GPU, i.e.
                ,batch size of each GPU.
            workers_per_gpu (int): How many subprocesses to use for data
                loading for each GPU.
            num_gpus (int): Number of GPUs. Only used in non-distributed
                training.
            dist (bool): Distributed training/test or not. Defaults to `False`.
            shuffle (bool): Whether to shuffle the data at every epoch.
                Defaults to `False`.
            seed (int): An integer set to be seed. Default is `None`.
            drop_last (bool): Whether to drop the last incomplete batch in
                epoch. Default to `False`.
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
        from mmdet.datasets import build_dataloader as build_dataloader_mmdet
        return build_dataloader_mmdet(
            dataset,
            samples_per_gpu,
            workers_per_gpu,
            num_gpus=num_gpus,
            dist=dist,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs)

    @staticmethod
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
        from mmdet.apis import single_gpu_test
        outputs = single_gpu_test(model, data_loader, show, out_dir, **kwargs)
        return outputs
