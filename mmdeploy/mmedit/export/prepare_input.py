from typing import Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmedit.datasets import build_dataloader as build_dataloader_mmedit
from mmedit.datasets import build_dataset as build_dataset_mmedit
from mmedit.datasets.pipelines import Compose
from torch.utils.data.dataset import Dataset

from mmdeploy.utils import Task, load_config


def _preprocess_cfg(config: Union[str, mmcv.Config], task: Task,
                    load_from_file: bool, is_static_cfg: bool,
                    input_shape: Sequence[int]):
    """Remove unnecessary information in config.

    Args:
        model_cfg (str | mmcv.Config): The input model config.
        task (Task): Specifying editing task type.
        load_from_file (bool): Whether the input is a filename of a numpy
            matrix. If this variable is True, extra preprocessing is required.
        is_static_cfg (bool): Whether the config specifys a static export.
            If this variable if True, the input image will be resize to a fix
            resolution.
        input_shape (Sequence[int]): A list of two integer in (width, height)
            format specifying input shape. Defaults to `None`.
    """

    # TODO: Differentiate the editing tasks (e.g. restorers and mattors
    # preprocess the data in differenet ways)

    if task == Task.SUPER_RESOLUTION:
        keys_to_remove = ['gt', 'gt_path']
    else:
        raise NotImplementedError(f'Unknown task type: {task.value}')

    # MMEdit doesn't support LoadImageFromWebcam.
    # Remove "LoadImageFromFile" and related metakeys.
    if not load_from_file:
        config.test_pipeline.pop(0)
        if task == Task.SUPER_RESOLUTION:
            keys_to_remove.append('lq_path')

    # Fix the input shape by 'Resize'
    if is_static_cfg:
        if task == Task.SUPER_RESOLUTION:
            resize = {
                'type': 'Resize',
                'scale': (input_shape[0], input_shape[1]),
                'keys': ['lq']
            }
            config.test_pipeline.insert(1, resize)

    for key in keys_to_remove:
        for pipeline in list(config.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                config.test_pipeline.remove(pipeline)
            if 'keys' in pipeline:
                while key in pipeline['keys']:
                    pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    config.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline:
                while key in pipeline['meta_keys']:
                    pipeline['meta_keys'].remove(key)


def create_input(task: Task,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Union[str, np.ndarray],
                 input_shape: Optional[Sequence[int]] = None,
                 device: Optional[str] = 'cuda:0'):
    """Create input for editing processor.

    Args:
        task (Task): Specifying editing task type.
        model_cfg (str | mmcv.Config): The input model config.
        imgs (str | np.ndarray): Input image(s).
        input_shape (Sequence[int]): A list of two integer in (width, height)
            format specifying input shape. Defaults to `None`.
        device (str): A string represents device type. Default is 'cuda:0'.

    Returns:
        tuple: (data, img), meta information for the input image and input.
    """
    if isinstance(imgs, (list, tuple)):
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')
    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    cfg = load_config(model_cfg)[0].copy()

    _preprocess_cfg(
        cfg,
        task=task,
        load_from_file=isinstance(imgs[0], str),
        is_static_cfg=input_shape is not None,
        input_shape=input_shape)

    test_pipeline = Compose(cfg.test_pipeline)

    data_arr = []
    for img in imgs:
        # TODO: This is only for restore. Add condiction statement.
        if isinstance(img, np.ndarray):
            data = dict(lq=img)
        else:
            data = dict(lq_path=img)

        data = test_pipeline(data)
        data_arr.append(data)

    data = collate(data_arr, samples_per_gpu=len(imgs))

    # TODO: This is only for restore. Add condiction statement.
    data['img'] = data['lq']

    if device != 'cpu':
        data = scatter(data, [device])[0]

    return data, data['img']


def build_dataset(dataset_cfg: Union[str, mmcv.Config], **kwargs):
    """Build dataset for processor.

    Args:
        dataset_cfg (str | mmcv.Config): The input dataset config.

    Returns:
        Dataset: A PyTorch dataset.
    """
    dataset_cfg = load_config(dataset_cfg)[0]
    data = dataset_cfg.data

    dataset = build_dataset_mmedit(data.test)
    return dataset


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
                     **kwargs):
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
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        kwargs (dict, optional): Any keyword argument to be used to initialize
            DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    return build_dataloader_mmedit(dataset, samples_per_gpu, workers_per_gpu,
                                   num_gpus, dist, shuffle, seed, drop_last,
                                   pin_memory, persistent_workers, **kwargs)
