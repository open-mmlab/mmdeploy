from typing import Any, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.parallel import DataContainer, collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmocr.datasets import build_dataloader as build_dataloader_mmocr
from mmocr.datasets import build_dataset as build_dataset_mmocr
from torch.utils.data import Dataset

from mmdeploy.utils import Task, load_config


def create_input(task: Task,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 input_shape: Sequence[int] = None,
                 device: str = 'cuda:0'):
    """Create input for text detector/recognizer.

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
    if isinstance(imgs, (list, tuple)):
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')

    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    if model_cfg.data.test['type'] == 'ConcatDataset':
        model_cfg.data.test.pipeline = \
            model_cfg.data.test['datasets'][0].pipeline

    is_ndarray = isinstance(imgs[0], np.ndarray)

    if is_ndarray:
        model_cfg = model_cfg.copy()
        # set loading pipeline type
        model_cfg.data.test.pipeline[0].type = 'LoadImageFromNdarray'

    test_pipeline = model_cfg.data.test.pipeline
    test_pipeline = replace_ImageToTensor(test_pipeline)
    # for static exporting
    if input_shape is not None:
        if task == Task.TEXT_DETECTION:
            test_pipeline[1].img_scale = tuple(input_shape)
            test_pipeline[1].transforms[0].keep_ratio = False
            test_pipeline[1].transforms[0].img_scale = tuple(input_shape)
        elif task == Task.TEXT_RECOGNITION:
            resize = {
                'height': input_shape[1],
                'min_width': input_shape[0],
                'max_width': input_shape[0],
                'keep_aspect_ratio': False
            }
            if 'transforms' in test_pipeline[1]:
                if test_pipeline[1].transforms[0].type == 'ResizeOCR':
                    test_pipeline[1].transforms[0].height = input_shape[1]
                    test_pipeline[1].transforms[0].max_width = input_shape[0]
                else:
                    raise ValueError(
                        f'Transforms[0] should be ResizeOCR, but got\
                         {test_pipeline[1].transforms[0].type}')
            else:
                test_pipeline[1].update(resize)
    from mmdet.datasets.pipelines import Compose
    from mmocr.datasets import build_dataset  # noqa: F401
    test_pipeline = Compose(test_pipeline)

    data_list = []
    for img in imgs:
        # prepare data
        if is_ndarray:
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)

        # build the data pipeline
        data = test_pipeline(data)
        # get tensor from list to stack for batch mode (text detection)
        data_list.append(data)

    if isinstance(data_list[0]['img'], list) and len(data_list) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(data_list)}')

    data = collate(data_list, samples_per_gpu=len(imgs))

    # process img_metas
    if isinstance(data['img_metas'], list):
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
    else:
        data['img_metas'] = data['img_metas'].data

    if isinstance(data['img'], list):
        data['img'] = [img.data for img in data['img']]
        if isinstance(data['img'][0], list):
            data['img'] = [img[0] for img in data['img']]
    else:
        data['img'] = data['img'].data

    if device != 'cpu':
        data = scatter(data, [device])[0]

    return data, data['img']


def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                  dataset_type: str = 'val',
                  **kwargs):
    """Build dataset for detector/recognizer.

    Args:
        dataset_cfg (str | mmcv.Config): The input dataset config.
        dataset_type (str): A string represents dataset type, e.g.: 'train',
            'test', 'val'. Defaults to 'val'.

    Returns:
        Dataset: A PyTorch dataset.
    """
    dataset_cfg = load_config(dataset_cfg)[0].copy()

    data = dataset_cfg.data
    assert dataset_type in data
    dataset = build_dataset_mmocr(data[dataset_type])

    return dataset


def build_dataloader(dataset: Dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus: int = 1,
                     dist: bool = False,
                     shuffle: bool = False,
                     seed: Optional[int] = None,
                     **kwargs):
    """Build dataloader for detector/recognizer.

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
    return build_dataloader_mmocr(
        dataset,
        samples_per_gpu,
        workers_per_gpu,
        num_gpus=num_gpus,
        dist=dist,
        shuffle=shuffle,
        seed=seed,
        **kwargs)


def get_tensor_from_input(input_data: tuple):
    """Get input tensor from input data.

    Args:
        input_data (tuple): Input data containing meta info and image tensor.
    Returns:
        torch.Tensor: An image in `Tensor`.
    """
    if isinstance(input_data['img'], DataContainer):
        return input_data['img'].data[0]
    return input_data['img'][0]
