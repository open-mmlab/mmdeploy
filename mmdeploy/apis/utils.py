import importlib
from typing import Any, Dict, Optional, Union

import mmcv


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
            raise ImportError(f'Can not import module: {codebase}')
    elif codebase == 'mmdet':
        if module_exist(codebase):
            from mmdet.apis import init_detector
            model = init_detector(model_cfg, model_checkpoint, device,
                                  cfg_options)
        else:
            raise ImportError(f'Can not import module: {codebase}')
    elif codebase == 'mmseg':
        if module_exist(codebase):
            from mmseg.apis import init_segmentor
            model = init_segmentor(model_cfg, model_checkpoint, device)
        else:
            raise ImportError(f'Can not import module: {codebase}')
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')

    return model


def create_input(codebase: str,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 device: str = 'cuda:0'):
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)
    elif not isinstance(model_cfg, (mmcv.Config, mmcv.ConfigDict)):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(model_cfg)}')

    cfg = model_cfg.copy()
    if codebase == 'mmcls':
        if module_exist(codebase):
            from mmdeploy.mmcls.export import create_input
            return create_input(cfg, imgs, device)
        else:
            raise ImportError(f'Can not import module: {codebase}')
    elif codebase == 'mmdet':
        if module_exist(codebase):
            from mmdeploy.mmdet.export import create_input
            return create_input(cfg, imgs, device)
        else:
            raise ImportError(f'Can not import module: {codebase}')
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')


def attribute_to_dict(attr):
    from onnx.helper import get_attribute_value
    ret = {}
    for a in attr:
        value = get_attribute_value(a)
        if isinstance(value, bytes):
            value = str(value, 'utf-8')
        ret[a.name] = value
    return ret
