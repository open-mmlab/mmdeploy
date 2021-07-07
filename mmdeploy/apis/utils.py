import importlib
from typing import Any, Dict, Optional, Union

import mmcv
import numpy as np
from mmcv.parallel import collate, scatter


def module_exist(module_name: str):
    return importlib.util.find_spec(module_name) is not None


def init_model(codebase: str,
               model_cfg: Union[str, mmcv.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[Dict] = None):
    # mmcls
    if codebase == 'mmcls':
        if module_exist(codebase):
            from mmcls.apis import init_model
            model = init_model(model_cfg, model_checkpoint, device,
                               cfg_options)
        else:
            raise ImportError('Can not import module: {}'.format(codebase))
    elif codebase == 'mmdet':
        if module_exist(codebase):
            from mmdet.apis import init_detector
            model = init_detector(model_cfg, model_checkpoint, device,
                                  cfg_options)
        else:
            raise ImportError('Can not import module: {}'.format(codebase))
    elif codebase == 'mmseg':
        if module_exist(codebase):
            from mmseg.apis import init_segmentor
            model = init_segmentor(model_cfg, model_checkpoint, device)
        else:
            raise ImportError('Can not import module: {}'.format(codebase))
    else:
        raise NotImplementedError('Unknown codebase type: {}'.format(codebase))

    return model


def create_input(codebase: str,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 device: str = 'cuda:0'):
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)
    elif not isinstance(model_cfg, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(model_cfg)}')

    cfg = model_cfg.copy()
    if codebase == 'mmcls':
        from mmcls.datasets.pipelines import Compose
        if isinstance(imgs, str):
            if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
                cfg.data.test.pipeline.insert(0,
                                              dict(type='LoadImageFromFile'))
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

    elif codebase == 'mmdet':
        if module_exist(codebase):
            from mmdet.datasets import replace_ImageToTensor
            from mmdet.datasets.pipelines import Compose

            if not isinstance(imgs, (list, tuple)):
                imgs = [imgs]

            if isinstance(imgs[0], np.ndarray):
                cfg = cfg.copy()
                # set loading pipeline type
                cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
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

            data['img_metas'] = [
                img_metas.data[0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]
            if device != 'cpu':
                data = scatter(data, [device])[0]

            return data, data['img']
        else:
            raise ImportError('Can not import module: {}'.format(codebase))
    else:
        raise NotImplementedError('Unknown codebase type: {}'.format(codebase))


def attribute_to_dict(attr):
    from onnx.helper import get_attribute_value
    ret = {}
    for a in attr:
        value = get_attribute_value(a)
        if isinstance(value, bytes):
            value = str(value, 'utf-8')
        ret[a.name] = value
    return ret