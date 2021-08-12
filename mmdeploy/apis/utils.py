import importlib
from typing import Any, Dict, Optional, Sequence, Union

import mmcv
import numpy as np
import torch


def assert_cfg_valid(cfg: Union[str, mmcv.Config, mmcv.ConfigDict], *args):
    """Check config validation."""

    def _assert_cfg_valid_(cfg):
        if isinstance(cfg, str):
            cfg = mmcv.Config.fromfile(cfg)
        if not isinstance(cfg, (mmcv.Config, mmcv.ConfigDict)):
            raise TypeError('deploy_cfg must be a filename or Config object, '
                            f'but got {type(cfg)}')

    _assert_cfg_valid_(cfg)
    for cfg in args:
        _assert_cfg_valid_(cfg)


def assert_module_exist(module_name: str):
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(f'Can not import module: {module_name}')


def init_model(codebase: str,
               model_cfg: Union[str, mmcv.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[Dict] = None):
    # mmcls
    if codebase == 'mmcls':
        assert_module_exist(codebase)
        from mmcls.apis import init_model
        model = init_model(model_cfg, model_checkpoint, device, cfg_options)

    elif codebase == 'mmdet':
        assert_module_exist(codebase)
        from mmdet.apis import init_detector
        model = init_detector(model_cfg, model_checkpoint, device, cfg_options)

    elif codebase == 'mmseg':
        assert_module_exist(codebase)
        from mmseg.apis import init_segmentor
        model = init_segmentor(model_cfg, model_checkpoint, device)

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
        assert_module_exist(codebase)
        from mmdeploy.mmcls.export import create_input
        return create_input(cfg, imgs, device)

    elif codebase == 'mmdet':
        assert_module_exist(codebase)
        from mmdeploy.mmdet.export import create_input
        return create_input(cfg, imgs, device)

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
        assert_module_exist(codebase)
        if backend == 'onnxruntime':
            from mmdeploy.mmcls.export import ONNXRuntimeClassifier
            backend_model = ONNXRuntimeClassifier(
                model_files[0], class_names=class_names, device_id=device_id)
        elif backend == 'tensorrt':
            from mmdeploy.mmcls.export import TensorRTClassifier
            backend_model = TensorRTClassifier(
                model_files[0], class_names=class_names, device_id=device_id)
        elif backend == 'ncnn':
            from mmdeploy.mmcls.export import NCNNClassifier
            backend_model = NCNNClassifier(
                model_files[0],
                model_files[1],
                class_names=class_names,
                device_id=device_id)
        elif backend == 'ppl':
            from mmdeploy.mmcls.export import PPLClassifier
            backend_model = PPLClassifier(
                model_files[0], class_names=class_names, device_id=device_id)
        else:
            raise NotImplementedError(f'Unsupported backend type: {backend}')
        return backend_model

    elif codebase == 'mmdet':
        assert_module_exist(codebase)
        if backend == 'onnxruntime':
            from mmdeploy.mmdet.export import ONNXRuntimeDetector
            backend_model = ONNXRuntimeDetector(
                model_files[0], class_names=class_names, device_id=device_id)
        elif backend == 'tensorrt':
            from mmdeploy.mmdet.export import TensorRTDetector
            backend_model = TensorRTDetector(
                model_files[0], class_names=class_names, device_id=device_id)
        elif backend == 'ppl':
            from mmdeploy.mmdet.export import PPLDetector
            backend_model = PPLDetector(
                model_files[0], class_names=class_names, device_id=device_id)
        else:
            raise NotImplementedError(f'Unsupported backend type: {backend}')
        return backend_model

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')


def get_classes_from_config(codebase: str, model_cfg: Union[str, mmcv.Config]):
    model_cfg_str = model_cfg
    if codebase == 'mmcls':
        assert_module_exist(codebase)
        if isinstance(model_cfg, str):
            model_cfg = mmcv.Config.fromfile(model_cfg)
        elif not isinstance(model_cfg, (mmcv.Config, mmcv.ConfigDict)):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(model_cfg)}')

        from mmcls.datasets import DATASETS
        module_dict = DATASETS.module_dict
        data_cfg = model_cfg.data

        if 'train' in data_cfg:
            module = module_dict[data_cfg.train.type]
        elif 'val' in data_cfg:
            module = module_dict[data_cfg.val.type]
        elif 'test' in data_cfg:
            module = module_dict[data_cfg.test.type]
        else:
            raise RuntimeError(f'No dataset config found in: {model_cfg_str}')

        return module.CLASSES

    if codebase == 'mmdet':
        assert_module_exist(codebase)
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
            raise RuntimeError(f'No dataset config found in: {model_cfg_str}')

        return module.CLASSES

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
        assert_module_exist(codebase)
        output_file = None if show_result else output_file
        with torch.no_grad():
            scores = model(**model_inputs, return_loss=False)[0]
            pred_score = np.max(scores, axis=0)
            pred_label = np.argmax(scores, axis=0)
            result = {
                'pred_label': pred_label,
                'pred_score': float(pred_score)
            }
            result['pred_class'] = model.CLASSES[result['pred_label']]
            model.show_result(
                show_img,
                result,
                show=True,
                win_name=backend,
                out_file=output_file)

    elif codebase == 'mmdet':
        assert_module_exist(codebase)
        output_file = None if show_result else output_file
        score_thr = 0.3
        with torch.no_grad():
            results = model(**model_inputs, return_loss=False, rescale=True)[0]
            model.show_result(
                show_img,
                results,
                score_thr=score_thr,
                show=True,
                win_name=backend,
                out_file=output_file)

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')
