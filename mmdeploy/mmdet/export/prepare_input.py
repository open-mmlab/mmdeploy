from typing import Any, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmdet.datasets import build_dataloader as build_dataloader_mmdet
from mmdet.datasets import build_dataset as build_dataset_mmdet
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from mmdeploy.utils import Task, load_config


def create_input(task: Task,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 input_shape: Sequence[int] = None,
                 device: str = 'cuda:0'):
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
    datas = []
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
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))

    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if device != 'cpu':
        data = scatter(data, [device])[0]

    return data, data['img']


def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                  dataset_type: str = 'val',
                  **kwargs):
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


def build_dataloader(dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus: int = 1,
                     dist: bool = False,
                     shuffle: bool = False,
                     seed: Optional[int] = None,
                     **kwargs):
    return build_dataloader_mmdet(
        dataset,
        samples_per_gpu,
        workers_per_gpu,
        num_gpus=num_gpus,
        dist=dist,
        shuffle=shuffle,
        seed=seed,
        **kwargs)


def get_tensor_from_input(input_data):
    return input_data['img'][0]
