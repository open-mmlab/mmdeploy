from typing import Sequence, Union

import mmcv
import torch
from mmcls.datasets import DATASETS
from mmcls.models import BaseClassifier

from mmdeploy.utils.config_utils import Backend, get_backend, load_config


class DeployBaseClassifier(BaseClassifier):
    """Base Class of Wrapper for classifier's inference.

    Args:
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, class_names: Sequence[str], device_id: int):
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
    """Wrapper for classifier's inference with ONNXRuntime.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 device_id: int):
        super(ONNXRuntimeClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Run test inference.

        Args:
            imgs (torch.Tensor): Input tensor of the model.

        Returns:
            list[np.ndarray]: Predictions of a classifier.
        """
        input_data = imgs
        results = self.model({'input': input_data})[0]
        return list(results)


class TensorRTClassifier(DeployBaseClassifier):
    """Wrapper for classifier's inference with TensorRT.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 device_id: int):
        super(TensorRTClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.tensorrt import TRTWrapper
        model = TRTWrapper(model_file)

        self.model = model

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Run test inference.

        Args:
            imgs (torch.Tensor): Input tensor of the model.

        Returns:
            list[np.ndarray]: Predictions of a classifier.
        """
        input_data = imgs
        with torch.cuda.device(self.device_id), torch.no_grad():
            results = self.model({'input': input_data})['output']
        results = results.detach().cpu().numpy()

        return list(results)


class NCNNClassifier(DeployBaseClassifier):
    """Wrapper for classifier's inference with NCNN.

    Args:
        param_file (str): Path of parameter file.
        bin_file (str): Path of bin file.
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, param_file: str, bin_file: str,
                 class_names: Sequence[str], device_id: int):
        super(NCNNClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.ncnn import NCNNWrapper
        self.model = NCNNWrapper(param_file, bin_file, output_names=['output'])

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Run test inference.

        Args:
            imgs (torch.Tensor): Input tensor of the model.

        Returns:
            list[np.ndarray]: Predictions of a classifier.
        """
        results = self.model({'input': imgs})['output']
        results = results.detach().cpu().numpy()
        results_list = list(results)
        return results_list


class PPLClassifier(DeployBaseClassifier):
    """Wrapper for classifier's inference with PPL.

    Args:
        model_file (str): Path of input ONNX model file.
        class_names (Sequence[str]): A list of string specifying class names.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file, class_names, device_id):
        super(PPLClassifier, self).__init__(class_names, device_id)
        from mmdeploy.apis.ppl import PPLWrapper
        model = PPLWrapper(model_file=model_file, device_id=device_id)
        self.model = model
        self.CLASSES = class_names

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs):
        """Run test inference.

        Args:
            imgs (torch.Tensor): Input tensor of the model.

        Returns:
            list[np.ndarray]: Predictions of a classifier.
        """
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
    """Get class name from config.

    Args:
        model_cfg (str | mmcv.Config): Input model config file or
            Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
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


def build_classifier(model_files: Sequence[str], model_cfg: Union[str,
                                                                  mmcv.Config],
                     deploy_cfg: Union[str,
                                       mmcv.Config], device_id: int, **kwargs):
    """Build classifier for different backend.

    Args:
        model_files (list[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device_id (int): An integer represents device index.

    Returns:
        DeployBaseClassifier: Classifier for a configured backend.
    """
    model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)

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
