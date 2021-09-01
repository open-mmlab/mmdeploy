from typing import Union

import mmcv
import torch
from mmcls.datasets import DATASETS
from mmcls.models import BaseClassifier

from mmdeploy.utils.config_utils import Backend, get_backend, load_config


class DeployBaseClassifier(BaseClassifier):
    """Base Class of Wrapper for classifier's inference."""

    def __init__(self, class_names, device_id):
        super(DeployBaseClassifier, self).__init__()
        self.CLASSES = class_names
        self.device_id = device_id

    def simple_test(self, img, *args, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward_test(self, imgs, *args, **kwargs):
        raise NotImplementedError('This method is not implemented.')


class ONNXRuntimeClassifier(DeployBaseClassifier):
    """Wrapper for classifier's inference with ONNXRuntime."""

    def __init__(self, model_file, class_names, device_id):
        super(ONNXRuntimeClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs
        results = self.model({'input': input_data})[0]
        return list(results)


class TensorRTClassifier(DeployBaseClassifier):

    def __init__(self, model_file, class_names, device_id):
        super(TensorRTClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.tensorrt import TRTWrapper
        model = TRTWrapper(model_file)

        self.model = model

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs
        with torch.cuda.device(self.device_id), torch.no_grad():
            results = self.model({'input': input_data})['output']
        results = results.detach().cpu().numpy()

        return list(results)


class NCNNClassifier(DeployBaseClassifier):

    def __init__(self, param_file, bin_file, class_names, device_id):
        super(NCNNClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.ncnn import NCNNWrapper
        self.model = NCNNWrapper(param_file, bin_file, output_names=['output'])

    def forward_test(self, imgs, *args, **kwargs):
        results = self.model({'input': imgs})['output']
        results = results.detach().cpu().numpy()
        results_list = list(results)
        return results_list


class PPLClassifier(DeployBaseClassifier):
    """Wrapper for classifier's inference with PPL."""

    def __init__(self, model_file, class_names, device_id):
        super(PPLClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.ppl import PPLWrapper
        model = PPLWrapper(model_file=model_file, device_id=device_id)
        self.model = model
        self.CLASSES = class_names

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs
        results = self.model({'input': input_data})[0]

        return list(results)


ONNXRUNTIME_CLASSIFIER_MAP = dict(end2end=ONNXRuntimeClassifier)

TENSORRT_CLASSIFIER_MAP = dict(end2end=TensorRTClassifier)

PPL_CLASSIFIER_MAP = dict(end2end=PPLClassifier)

NCNN_CLASSIFIER_MAP = dict(end2end=NCNNClassifier)

BACKEND_CLASSIFIER_MAP = {
    Backend.ONNXRUNTIME: ONNXRUNTIME_CLASSIFIER_MAP,
    Backend.TENSORRT: TENSORRT_CLASSIFIER_MAP,
    Backend.PPL: PPL_CLASSIFIER_MAP,
    Backend.NCNN: NCNN_CLASSIFIER_MAP
}


def get_classes_from_config(model_cfg: Union[str, mmcv.Config]):
    model_cfg = load_config(model_cfg)[0]
    module_dict = DATASETS.module_dict
    data_cfg = model_cfg.data

    if 'train' in data_cfg:
        module = module_dict[data_cfg.train.type]
    elif 'val' in data_cfg:
        module = module_dict[data_cfg.val.type]
    elif 'test' in data_cfg:
        module = module_dict[data_cfg.test.type]
    else:
        raise RuntimeError(f'No dataset config found in: {model_cfg}')

    return module.CLASSES


def build_classifier(model_files, model_cfg, deploy_cfg, device_id, **kwargs):
    model_cfg = load_config(model_cfg)[0]
    deploy_cfg = load_config(deploy_cfg)[0]

    backend = get_backend(deploy_cfg)
    class_names = get_classes_from_config(model_cfg)

    assert backend in BACKEND_CLASSIFIER_MAP, \
        f'Unsupported backend type: {backend.value}'
    model_map = BACKEND_CLASSIFIER_MAP[backend]

    model_type = 'end2end'
    assert model_type in model_map, f'Unsupported model type: {model_type}'
    backend_classifier_class = model_map[model_type]

    backend_detector = backend_classifier_class(
        *model_files, class_names=class_names, device_id=device_id)

    return backend_detector
