from typing import Any, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmseg.apis.inference import LoadImage
from mmseg.datasets import build_dataloader as build_dataloader_mmseg
from mmseg.datasets import build_dataset as build_dataset_mmseg
from mmseg.datasets.pipelines import Compose

from mmdeploy.utils.config_utils import load_config


def create_input(model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 device: str = 'cuda:0'):

    cfg = load_config(model_cfg)[0].copy()
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    # TODO remove hard code
    cfg.data.test.pipeline[1]['img_scale'] = (1024, 512)
    cfg.data.test.pipeline[1]['transforms'][0]['keep_ratio'] = False
    cfg.data.test.pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]

    test_pipeline = Compose(cfg.data.test.pipeline)
    datas = []
    for img in imgs:
        # prepare data
        data = dict(img=img)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))

    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0][None, :] for img in data['img']]
    if device != 'cpu':
        data = scatter(data, [device])[0]

    return data, data['img']


def build_dataset(dataset_cfg: Union[str, mmcv.Config],
                  dataset_type: str = 'val',
                  **kwargs):
    dataset_cfg = load_config(dataset_cfg)[0]
    data = dataset_cfg.data
    assert dataset_type in data

    dataset = build_dataset_mmseg(data[dataset_type])

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu: int,
                     workers_per_gpu: int,
                     num_gpus=1,
                     dist=False,
                     shuffle=False,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    return build_dataloader_mmseg(dataset, samples_per_gpu, workers_per_gpu,
                                  num_gpus, dist, shuffle, seed, drop_last,
                                  pin_memory, persistent_workers, **kwargs)


def get_tensor_from_input(input_data):
    return input_data['img']
