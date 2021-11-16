from typing import Sequence, Union

import mmcv
import numpy as np
import torch
from mmseg.datasets import DATASETS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize

from mmdeploy.utils.config_utils import Backend, get_backend, load_config


class DeployBaseSegmentor(BaseSegmentor):
    """Base Class of wrapper for segmentation's inference.

    Args:
        class_names (Sequence[str]): A list of string specifying class names.
        palette (np.ndarray): The palette of segmentation map.
        device_id (int): An integer represents device index.
    """

    def __init__(self, class_names: Sequence[str], palette: np.ndarray,
                 device_id: int):
        super(DeployBaseSegmentor, self).__init__(init_cfg=None)
        self.CLASSES = class_names
        self.device_id = device_id
        self.PALETTE = palette

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def encode_decode(self, img, img_metas):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def simple_test(self, img, img_meta, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward(self, img: Sequence[torch.Tensor], img_metas: Sequence[dict],
                **kwargs):
        """Run forward test.

        Args:
            img (Sequence[torch.Tensor]]): A list of input image tensor(s).
            img_metas (Sequence[dict]]): A list of dict containing image(s)
                meta information.

        Returns:
            list[np.ndarray]: A list of segmentation result.
        """
        if isinstance(img, (list, tuple)):
            img = img[0]
        img = img.contiguous()
        seg_pred = self.forward_test(img, img_metas, **kwargs)
        # whole mode supports dynamic shape
        ori_shape = img_metas[0][0]['ori_shape']
        if not (ori_shape[0] == seg_pred.shape[-2]
                and ori_shape[1] == seg_pred.shape[-1]):
            seg_pred = torch.from_numpy(seg_pred).float()
            seg_pred = resize(
                seg_pred, size=tuple(ori_shape[:2]), mode='nearest')
            seg_pred = seg_pred.long().detach().cpu().numpy()
        # remove unnecessary dim
        seg_pred = seg_pred.squeeze(1)
        seg_pred = list(seg_pred)
        return seg_pred


class ONNXRuntimeSegmentor(DeployBaseSegmentor):
    """Wrapper for segmentation's inference with ONNX Runtime.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        palette (np.ndarray): The palette of segmentation map.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 palette: np.ndarray, device_id: int):
        super(ONNXRuntimeSegmentor, self).__init__(class_names, palette,
                                                   device_id)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_test(self, imgs: torch.Tensor, img_metas: Sequence[dict],
                     **kwargs):
        """Run forward test to get predictions.

        Args:
            imgs (torch.Tensor): Input tensor of the model.
            img_metas (Sequence[dict]]): A list of dict containing image(s)
                meta information.
        Returns:
            torch.Tensor: Segmentation result.
        """
        seg_pred = self.model({'input': imgs})[0]
        return seg_pred


class TensorRTSegmentor(DeployBaseSegmentor):
    """Wrapper for segmentation's inference with TensorRT.

    Args:
        model_file (str): The path of input model file.
        class_names (Sequence[str]): A list of string specifying class names.
        palette (np.ndarray): The palette of segmentation map.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 palette: np.ndarray, device_id: int):
        super(TensorRTSegmentor, self).__init__(class_names, palette,
                                                device_id)
        from mmdeploy.apis.tensorrt import TRTWrapper

        model = TRTWrapper(model_file)
        self.model = model
        self.output_name = self.model.output_names[0]

    def forward_test(self, imgs: torch.Tensor, img_metas: Sequence[dict],
                     **kwargs):
        """Run forward test to get predictions.

        Args:
            imgs (torch.Tensor): Input tensor of the model.
            img_metas (Sequence[dict]]): A list of dict containing image(s)
                meta information.
        Returns:
            np.ndarray: Segmentation result.
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            seg_pred = self.model({'input': imgs})[self.output_name]
        seg_pred = seg_pred.detach().cpu().numpy()
        return seg_pred


class PPLSegmentor(DeployBaseSegmentor):
    """Wrapper for segmentation's inference with PPL.

    Args:
        model_file (Sequence[str]): Paths of input params and bin files.
        class_names (Sequence[str]): A list of string specifying class names.
        palette (np.ndarray): The palette of segmentation map.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: str, class_names: Sequence[str],
                 palette: np.ndarray, device_id: int):
        super(PPLSegmentor, self).__init__(class_names, palette, device_id)
        from mmdeploy.apis.ppl import PPLWrapper
        self.model = PPLWrapper(model_file[0], model_file[1], device_id)

    def forward_test(self, imgs: torch.Tensor, img_metas: Sequence[dict],
                     **kwargs):
        """Run forward test to get predictions.

        Args:
            imgs (torch.Tensor): Input tensor of the model.
            img_metas (Sequence[dict]]): A list of dict containing image(s)
                meta information.
        Returns:
            np.ndarray: Segmentation result.
        """
        seg_pred = self.model({'input': imgs})[0]
        return seg_pred


class NCNNSegmentor(DeployBaseSegmentor):
    """Wrapper for segmentation's inference with NCNN.

    Args:
        model_file (Sequence[str]): Paths of input params and bin files.
        class_names (Sequence[str]): A list of string specifying class names.
        palette (np.ndarray): The palette of segmentation map.
        device_id (int): An integer represents device index.
    """

    def __init__(self, model_file: Sequence[str], class_names: Sequence[str],
                 palette: np.ndarray, device_id: int):
        super(NCNNSegmentor, self).__init__(class_names, palette, device_id)
        from mmdeploy.apis.ncnn import NCNNWrapper
        assert len(model_file) == 2, f'`model_file` should be [param_file, \
            bin_file], but given {model_file}'

        ncnn_param_file = model_file[0]
        ncnn_bin_file = model_file[1]
        self.model = NCNNWrapper(
            ncnn_param_file, ncnn_bin_file, output_names=['output'])

    def forward_test(self, imgs: torch.Tensor, img_metas: Sequence[dict],
                     **kwargs):
        """Run forward test to get predictions.

        Args:
            imgs (torch.Tensor): Input tensor of the model.
            img_metas (Sequence[dict]]): A list of dict containing image(s)
                meta information.
        Returns:
            np.ndarray: Segmentation result.
        """
        results = self.model({'input': imgs})['output']
        results = results.detach().cpu().numpy()
        return results


ONNXRUNTIME_SEGMENTOR_MAP = dict(end2end=ONNXRuntimeSegmentor)

TENSORRT_SEGMENTOR_MAP = dict(end2end=TensorRTSegmentor)

PPL_SEGMENTOR_MAP = dict(end2end=PPLSegmentor)
NCNN_SEGMENTOR_MAP = dict(end2end=NCNNSegmentor)

BACKEND_SEGMENTOR_MAP = {
    Backend.ONNXRUNTIME: ONNXRUNTIME_SEGMENTOR_MAP,
    Backend.TENSORRT: TENSORRT_SEGMENTOR_MAP,
    Backend.PPL: PPL_SEGMENTOR_MAP,
    Backend.NCNN: NCNN_SEGMENTOR_MAP
}


def get_classes_palette_from_config(model_cfg: Union[str, mmcv.Config]):
    """Get class name and palette from config.

    Args:
        model_cfg (str | mmcv.Config): Input model config file or
            Config object.

    Returns:
        tuple(Sequence[str], np.ndarray): A list of string specifying names of
            different class and the palette of segmentation map.
    """
    # load cfg if necessary
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

    return module.CLASSES, module.PALETTE


def build_segmentor(model_files, model_cfg, deploy_cfg, device_id):
    """Build segmentor for different backend.

    Args:
        model_files (list[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device_id (int): An integer represents device index.

    Returns:
        DeployBaseSegmentor: Segmentor for a configured backend.
    """
    # load cfg if necessary
    model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)

    backend = get_backend(deploy_cfg)
    class_names, palette = get_classes_palette_from_config(model_cfg)
    assert backend in BACKEND_SEGMENTOR_MAP, \
        f'Unsupported backend type: {backend.value}'
    segmentor_map = BACKEND_SEGMENTOR_MAP[backend]

    model_type = 'end2end'
    assert model_type in segmentor_map, f'Unsupported model type: {model_type}'
    backend_segmentor_class = segmentor_map[model_type]
    model_files = model_files[0] if len(model_files) == 1 else model_files
    backend_segmentor = backend_segmentor_class(
        model_files,
        class_names=class_names,
        device_id=device_id,
        palette=palette)

    return backend_segmentor
