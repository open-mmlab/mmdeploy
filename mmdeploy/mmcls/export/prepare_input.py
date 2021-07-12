from typing import Any, Union

import mmcv
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


def create_input(model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 device: str = 'cuda:0'):
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)
    elif not isinstance(model_cfg, (mmcv.Config, mmcv.ConfigDict)):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(model_cfg)}')

    cfg = model_cfg.copy()
    if isinstance(imgs, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=imgs), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=imgs)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if device != 'cpu':
        data = scatter(data, [device])[0]
    return data, data['img']
