from typing import Any, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor


def create_input(model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 device: str = 'cuda:0'):
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

    model_cfg.data.test.pipeline = replace_ImageToTensor(
        model_cfg.data.test.pipeline)
    from mmdet.datasets.pipelines import Compose
    from mmocr.datasets import build_dataset  # noqa: F401
    test_pipeline = Compose(model_cfg.data.test.pipeline)

    datas = []
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
        datas.append(data)

    if isinstance(datas[0]['img'], list) and len(datas) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(datas)}')

    data = collate(datas, samples_per_gpu=len(imgs))

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
