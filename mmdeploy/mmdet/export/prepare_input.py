from typing import Any, Dict, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmdet.datasets import build_dataloader as build_dataloader_mmdet
from mmdet.datasets import build_dataset as build_dataset_mmdet
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

from mmdeploy.utils import Task, load_config


def create_input(task: Task,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 input_shape: Sequence[int] = None,
                 device: str = 'cuda:0'):
    """Create input for detector.

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
    assert task == Task.OBJECT_DETECTION
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
        transforms = cfg.data.test.pipeline[1]['transforms']
        for trans in transforms:
            trans_type = trans['type']
            if trans_type == 'Resize':
                trans['keep_ratio'] = False
            elif trans_type == 'Pad':
                trans['size_divisor'] = 1

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data_list = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        data_list.append(data)

    data = collate(data_list, samples_per_gpu=len(imgs))

    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if device != 'cpu':
        data = scatter(data, [device])[0]

    return data, data['img']


def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                  dataset_type: str = 'val',
                  **kwargs):
    """Build dataset for detection.

    Args:
        dataset_cfg (str | mmcv.Config): The input dataset config.
        dataset_type (str): A string represents dataset type, e.g.: 'train',
            'test', 'val'. Defaults to 'val'.

    Returns:
        Dataset: A PyTorch dataset.
    """
    dataset_cfg = load_config(dataset_cfg)[0].copy()

    assert dataset_type in dataset_cfg.data
    data_cfg = dataset_cfg.data[dataset_type]
    # in case the dataset is concatenated
    if isinstance(data_cfg, dict):
        data_cfg.test_mode = True
        samples_per_gpu = data_cfg.get('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            data_cfg.pipeline = replace_ImageToTensor(data_cfg.pipeline)
    elif isinstance(data_cfg, list):
        for ds_cfg in data_cfg:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.get('samples_per_gpu', 1) for ds_cfg in data_cfg])
        if samples_per_gpu > 1:
            for ds_cfg in data_cfg:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    dataset = build_dataset_mmdet(data_cfg)

    return dataset


def build_dataloader(dataset: Dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus: int = 1,
                     dist: bool = False,
                     shuffle: bool = False,
                     seed: Optional[int] = None,
                     **kwargs):
    """Build dataloader for detection.

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
        kwargs: Any other keyword argument to be used to initialize DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    return build_dataloader_mmdet(
        dataset,
        samples_per_gpu,
        workers_per_gpu,
        num_gpus=num_gpus,
        dist=dist,
        shuffle=shuffle,
        seed=seed,
        **kwargs)


def get_tensor_from_input(input_data: Dict[str, Any]):
    """Get input tensor from input data.

    Args:
        input_data (dict): Input data containing meta info and image tensor.
    Returns:
        torch.Tensor: An image in `Tensor`.
    """
    return input_data['img'][0]
