from typing import Any, Sequence, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmedit.datasets import build_dataloader as build_dataloadeer_mmedit
from mmedit.datasets import build_dataset as build_dataset_mmedit
from mmedit.datasets.pipelines import Compose

from mmdeploy.utils import Task, load_config


def _preprocess_cfg(config):
    # TODO: Differentiate the editting tasks (e.g. restorers and mattors
    # preprocess the data in differenet ways)

    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(config.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                config.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    config.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)


def create_input(task: Task,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 input_shape: Sequence[int] = None,
                 device: str = 'cuda:0'):
    if isinstance(imgs, (list, tuple)):
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')
    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    cfg = load_config(model_cfg)[0].copy()
    _preprocess_cfg(cfg)

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.test_pipeline[0].type = 'LoadImageFromWebcam'

    # for static exporting
    if input_shape is not None:
        if task == Task.SUPER_RESOLUTION:
            resize = {
                'type': 'Resize',
                'scale': (input_shape[0], input_shape[1]),
                'keys': ['lq']
            }
            cfg.test_pipeline.insert(1, resize)
        else:
            raise NotImplementedError(f'Unknown task type: {task.value}')

    test_pipeline = Compose(cfg.test_pipeline)

    data_arr = []
    for img in imgs:
        # TODO: This is only for restore. Add condiction statement
        data = dict(lq_path=img)

        data = test_pipeline(data)
        data_arr.append(data)

    data = collate(data_arr, samples_per_gpu=len(imgs))

    # TODO: This is only for restore. Add condiction statement
    data['img'] = data['lq']

    if device != 'cpu':
        data = scatter(data, [device])[0]

    return data, data['img']


def build_dataset(dataset_cfg: Union[str, mmcv.Config], **kwargs):
    dataset_cfg = load_config(dataset_cfg)[0]
    data = dataset_cfg.data

    dataset = build_dataset_mmedit(data.test)
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

    return build_dataloadeer_mmedit(dataset, samples_per_gpu, workers_per_gpu,
                                    num_gpus, dist, shuffle, seed, drop_last,
                                    pin_memory, persistent_workers, **kwargs)
