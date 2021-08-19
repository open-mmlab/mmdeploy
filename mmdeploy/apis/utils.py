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
        return cfg

    args = (cfg, ) + args
    ret = [_assert_cfg_valid_(cfg) for cfg in args]

    return ret


def assert_module_exist(module_name: str):
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(f'Can not import module: {module_name}')


def init_model(codebase: str,
               model_cfg: Union[str, mmcv.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[Dict] = None):
    assert_module_exist(codebase)
    if codebase == 'mmcls':
        from mmcls.apis import init_model
        model = init_model(model_cfg, model_checkpoint, device, cfg_options)

    elif codebase == 'mmdet':
        from mmdet.apis import init_detector
        model = init_detector(model_cfg, model_checkpoint, device, cfg_options)

    elif codebase == 'mmseg':
        from mmseg.apis import init_segmentor
        from mmdeploy.mmseg.export import convert_syncbatchnorm
        model = init_segmentor(model_cfg, model_checkpoint, device)
        model = convert_syncbatchnorm(model)

    elif codebase == 'mmocr':
        from mmdet.apis import init_detector
        from mmocr.models import build_detector  # noqa: F401
        model = init_detector(model_cfg, model_checkpoint, device, cfg_options)

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')

    return model.eval()


def create_input(codebase: str,
                 model_cfg: Union[str, mmcv.Config],
                 imgs: Any,
                 device: str = 'cuda:0'):
    model_cfg = assert_cfg_valid(model_cfg)[0]

    assert_module_exist(codebase)
    cfg = model_cfg.copy()
    if codebase == 'mmcls':
        from mmdeploy.mmcls.export import create_input
        return create_input(cfg, imgs, device)

    elif codebase == 'mmdet':
        from mmdeploy.mmdet.export import create_input
        return create_input(cfg, imgs, device)

    elif codebase == 'mmocr':
        from mmdeploy.mmocr.export import create_input
        return create_input(cfg, imgs, device)

    elif codebase == 'mmseg':
        from mmdeploy.mmseg.export import create_input
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
                       model_cfg: Union[str, mmcv.Config],
                       deploy_cfg: Union[str, mmcv.Config],
                       device_id: int = 0,
                       **kwargs):
    deploy_cfg, model_cfg = assert_cfg_valid(deploy_cfg, model_cfg)

    codebase = deploy_cfg['codebase']
    backend = deploy_cfg['backend']
    assert_module_exist(codebase)
    if codebase != 'mmocr':
        class_names = get_classes_from_config(codebase, model_cfg)

    if codebase == 'mmcls':
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
        from mmdeploy.mmdet.export.model_wrappers import build_detector
        return build_detector(
            model_files, model_cfg, deploy_cfg, device_id=device_id)

    elif codebase == 'mmseg':
        if backend == 'onnxruntime':
            from mmdeploy.mmseg.export import ONNXRuntimeSegmentor
            backend_model = ONNXRuntimeSegmentor(
                model_files[0], class_names=class_names, device_id=device_id)
        elif backend == 'tensorrt':
            from mmdeploy.mmseg.export import TensorRTSegmentor
            backend_model = TensorRTSegmentor(
                model_files[0], class_names=class_names, device_id=device_id)
        else:
            raise NotImplementedError(f'Unsupported backend type: {backend}')
        return backend_model

    elif codebase == 'mmocr':
        algorithm_type = deploy_cfg['algorithm_type']
        if backend == 'onnxruntime':
            if algorithm_type == 'det':
                from mmdeploy.mmocr.export import ONNXRuntimeDetector
                backend_model = ONNXRuntimeDetector(
                    model_files[0], cfg=model_cfg, device_id=device_id)
            elif algorithm_type == 'recog':
                from mmdeploy.mmocr.export import ONNXRuntimeRecognizer
                backend_model = ONNXRuntimeRecognizer(
                    model_files[0], cfg=model_cfg, device_id=device_id)
        elif backend == 'tensorrt':
            if algorithm_type == 'det':
                from mmdeploy.mmocr.export import TensorRTDetector
                backend_model = TensorRTDetector(
                    model_files[0], cfg=model_cfg, device_id=device_id)
            elif algorithm_type == 'recog':
                from mmdeploy.mmocr.export import TensorRTRecognizer
                backend_model = TensorRTRecognizer(
                    model_files[0], cfg=model_cfg, device_id=device_id)
        else:
            raise NotImplementedError(f'Unsupported backend type: {backend}')
        return backend_model

    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')


def get_classes_from_config(codebase: str, model_cfg: Union[str, mmcv.Config]):
    assert_module_exist(codebase)

    model_cfg_str = model_cfg
    model_cfg = assert_cfg_valid(model_cfg)[0]

    if codebase == 'mmdet':
        from mmdeploy.mmdet.export.model_wrappers \
            import get_classes_from_config as get_classes_mmdet
        return get_classes_mmdet(model_cfg)

    if codebase == 'mmcls':
        from mmcls.datasets import DATASETS
    elif codebase == 'mmdet':
        from mmdet.datasets import DATASETS
    elif codebase == 'mmseg':
        from mmseg.datasets import DATASETS
    elif codebase == 'mmocr':
        from mmocr.datasets import DATASETS
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')

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


def check_model_outputs(codebase: str,
                        image: Union[str, np.ndarray],
                        model_inputs,
                        model,
                        output_file: str,
                        backend: str,
                        dataset: str = None,
                        show_result=False):
    assert_module_exist(codebase)
    show_img = mmcv.imread(image) if isinstance(image, str) else image

    if codebase == 'mmcls':
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

    elif codebase == 'mmocr':
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

    elif codebase == 'mmseg':
        output_file = None if show_result else output_file
        from mmseg.core.evaluation import get_palette
        dataset = 'cityscapes' if dataset is None else dataset
        palette = get_palette(dataset)
        with torch.no_grad():
            results = model(**model_inputs, return_loss=False, rescale=True)
            model.show_result(
                show_img,
                results,
                palette=palette,
                show=True,
                win_name=backend,
                out_file=output_file,
                opacity=0.5)
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')


def get_split_cfg(codebase: str, split_type: str):
    if codebase == 'mmdet':
        assert_module_exist(codebase)
        from mmdeploy.mmdet.export import get_split_cfg \
            as get_split_cfg_mmdet
        return get_split_cfg_mmdet(split_type)
    else:
        raise NotImplementedError(f'Unknown codebase type: {codebase}')
