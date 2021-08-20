from typing import Any, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose

from mmdeploy.apis.utils import load_config


def create_input(model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 device: str = 'cuda:0'):

    cfg = load_config(model_cfg).copy()
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
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
