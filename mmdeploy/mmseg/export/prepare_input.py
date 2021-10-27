from typing import Any, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmseg.apis.inference import LoadImage
from mmseg.datasets import build_dataloader as build_dataloader_mmseg
from mmseg.datasets import build_dataset as build_dataset_mmseg
from mmseg.datasets.pipelines import Compose
from torch.utils.data import Dataset

from mmdeploy.utils import Task, load_config


def create_input(task: Task,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 input_shape: Optional[Sequence[int]] = None,
                 device: str = 'cuda:0'):
    """Create input for segmentation.

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
    assert task == Task.SEGMENTATION
    cfg = load_config(model_cfg)[0].copy()
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    # for static exporting
    if input_shape is not None:
        cfg.data.test.pipeline[1]['img_scale'] = tuple(input_shape)
    cfg.data.test.pipeline[1]['transforms'][0]['keep_ratio'] = False
    cfg.data.test.pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]

    test_pipeline = Compose(cfg.data.test.pipeline)
    data_list = []
    for img in imgs:
        # prepare data
        data = dict(img=img)
        # build the data pipeline
        data = test_pipeline(data)
        data_list.append(data)

    data = collate(data_list, samples_per_gpu=len(imgs))

    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0][None, :] for img in data['img']]
    if device != 'cpu':
        data = scatter(data, [device])[0]

    return data, data['img']


def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                  dataset_type: str = 'val',
                  **kwargs):
    """Build dataset for segmentation.

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

    dataset = build_dataset_mmseg(data[dataset_type])

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
    """Build dataloader for segmentation.

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
        seed (int): An integer set to be seed. Default is `None`.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default to `False`.
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
    return build_dataloader_mmseg(dataset, samples_per_gpu, workers_per_gpu,
                                  num_gpus, dist, shuffle, seed, drop_last,
                                  pin_memory, persistent_workers, **kwargs)


def get_tensor_from_input(input_data: tuple):
    """Get input tensor from input data.

    Args:
        input_data (tuple): Input data containing meta info and image tensor.
    Returns:
        torch.Tensor: An image in `Tensor`.
    """
    return input_data['img'][0]
