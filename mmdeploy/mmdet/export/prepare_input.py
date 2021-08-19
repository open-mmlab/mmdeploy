from typing import Any, Optional, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmdet.datasets import build_dataloader as build_dataloader_mmdet
from mmdet.datasets import build_dataset as build_dataset_mmdet
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


def create_input(model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 device: str = 'cuda:0'):
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)
    elif not isinstance(model_cfg, (mmcv.Config, mmcv.ConfigDict)):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(model_cfg)}')
    cfg = model_cfg.copy()

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

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
    if isinstance(dataset_cfg, str):
        dataset_cfg = mmcv.Config.fromfile(dataset_cfg)
    elif not isinstance(dataset_cfg, (mmcv.Config, mmcv.ConfigDict)):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(dataset_cfg)}')

    data = dataset_cfg.data
    assert dataset_type in data

    dataset = build_dataset_mmdet(data[dataset_type])

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus: int = 1,
                     dist: bool = True,
                     shuffle: bool = True,
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
