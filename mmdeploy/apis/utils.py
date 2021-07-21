import importlib
from typing import Any, Dict, Optional, Sequence, Union

import mmcv
import numpy as np
import torch


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


def init_backend_model(model_files: Sequence[str],
                       codebase: str,
                       backend: str,
                       class_names: Sequence[str],
                       device_id: int = 0):
    if codebase == 'mmcls':
        if module_exist(codebase):
            raise NotImplementedError(f'Unsupported codebase type: {codebase}')
        else:
            raise ImportError(f'Can not import module: {codebase}')
    elif codebase == 'mmdet':
        if module_exist(codebase):
            if backend == 'onnxruntime':
                from mmdeploy.mmdet.export import ONNXRuntimeDetector
                backend_model = ONNXRuntimeDetector(
                    model_files[0],
                    class_names=class_names,
                    device_id=device_id)
            elif backend == 'tensorrt':
                from mmdeploy.mmdet.export import TensorRTDetector
                backend_model = TensorRTDetector(
                    model_files[0],
                    class_names=class_names,
                    device_id=device_id)
            else:
                raise NotImplementedError(
                    f'Unsupported backend type: {backend}')
            return backend_model
        else:
            raise ImportError(f'Can not import module: {codebase}')
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')


def get_classes_from_config(codebase: str, model_cfg: Union[str, mmcv.Config]):
    model_cfg_str = model_cfg
    if codebase == 'mmdet':
        if module_exist(codebase):
            if isinstance(model_cfg, str):
                model_cfg = mmcv.Config.fromfile(model_cfg)
            elif not isinstance(model_cfg, (mmcv.Config, mmcv.ConfigDict)):
                raise TypeError('config must be a filename or Config object, '
                                f'but got {type(model_cfg)}')

            from mmdet.datasets import DATASETS
            module_dict = DATASETS.module_dict
            data_cfg = model_cfg.data

            if 'train' in data_cfg:
                module = module_dict[data_cfg.train.type]
            elif 'val' in data_cfg:
                module = module_dict[data_cfg.val.type]
            elif 'test' in data_cfg:
                module = module_dict[data_cfg.test.type]
            else:
                raise RuntimeError(
                    f'No dataset config found in: {model_cfg_str}')

            return module.CLASSES
        else:
            raise ImportError(f'Can not import module: {codebase}')
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')


def check_model_outputs(codebase: str,
                        image: Union[str, np.ndarray],
                        model_inputs,
                        model,
                        output_file: str,
                        backend: str,
                        show_result=False):
    show_img = mmcv.imread(image) if isinstance(image, str) else image
    if codebase == 'mmcls':
        if module_exist(codebase):
            raise NotImplementedError(f'Unsupported codebase type: {codebase}')
        else:
            raise ImportError(f'Can not import module: {codebase}')
    elif codebase == 'mmdet':
        if module_exist(codebase):
            output_file = None if show_result else output_file
            score_thr = 0.3
            with torch.no_grad():
                results = model(
                    **model_inputs, return_loss=False, rescale=True)[0]
                model.show_result(
                    show_img,
                    results,
                    score_thr=score_thr,
                    show=True,
                    win_name=backend,
                    out_file=output_file)

        else:
            raise ImportError(f'Can not import module: {codebase}')
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')
