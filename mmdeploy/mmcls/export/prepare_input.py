import logging
from typing import Any, Optional, Sequence, Union

import mmcv
from mmcls.datasets import build_dataloader as build_dataloader_mmcls
from mmcls.datasets import build_dataset as build_dataset_mmcls
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from torch.utils.data import Dataset

from mmdeploy.utils import Task, load_config


def create_input(task: Task,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 input_shape: Optional[Sequence[int]] = None,
                 device: str = 'cuda:0'):
    """Create input for classifier.

    Args:
        task (Task): Specifying task type.
        model_cfg (str | mmcv.Config): The input model config.
        imgs (Any): Input image(s), accpeted data type are `str`,
            `np.ndarray`, `torch.Tensor`.
        input_shape (list[int]): A list of two integer in (width, height)
            format specifying input shape. Defaults to `None`.
        device (str): A string represents device type. Default is 'cuda:0'.

    Returns:
        tuple: (data, img), meta information for the input image and input.
    """
    assert task == Task.CLASSIFICATION
    cfg = load_config(model_cfg)[0].copy()
    if isinstance(imgs, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=imgs), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=imgs)
    # check whether input_shape is valid
    if input_shape is not None:
        if 'crop_size' in cfg.data.test.pipeline[2]:
            crop_size = cfg.data.test.pipeline[2]['crop_size']
            if tuple(input_shape) != (crop_size, crop_size):
                logging.warning(
                    f'`input shape` should be equal to `crop_size`: {crop_size},\
                     but given: {input_shape}')
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if device != 'cpu':
        data = scatter(data, [device])[0]
    return data, data['img']


def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                  dataset_type: str = 'val',
                  **kwargs):
    """Build dataset for classifier.

    Args:
        dataset_cfg (str | mmcv.Config): The input dataset config.
        dataset_type (str): A string represents dataset type, e.g.: 'train',
            'test', 'val'. Defaults to 'val'.

    Returns:
        Dataset: A PyTorch dataset.
    """
    dataset_cfg = load_config(dataset_cfg)[0]
    data = dataset_cfg.data
    assert dataset_type in data

    dataset = build_dataset_mmcls(data[dataset_type])

    return dataset


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
                     **kwargs):
    """Build dataloader for classifier.

    Args:
        dataset (Dataset): Input dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Defaults to `False`.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Defaults to `False`.
        round_up (bool): Whether to round up the length of dataset by adding
            extra samples to make it evenly divisible. Default is `True`.
        seed (int): An integer set to be seed. Default is `None`.
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default is `True`.
        persistent_workers (bool): If `True`, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default is `True`.
        kwargs: Any other keyword argument to be used to initialize DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    return build_dataloader_mmcls(dataset, samples_per_gpu, workers_per_gpu,
                                  num_gpus, dist, shuffle, round_up, seed,
                                  pin_memory, persistent_workers, **kwargs)


def get_tensor_from_input(input_data: tuple):
    """Get input tensor from input data.

    Args:
        input_data (tuple): Input data containing meta info and image tensor.
    Returns:
        torch.Tensor: An image in `Tensor`.
    """
    return input_data['img']
