from typing import Iterable, Sequence, Union

import mmcv
import torch
from mmdet.models.builder import DETECTORS
from mmocr.datasets import DATASETS
from mmocr.models.textdet.detectors import (SingleStageTextDetector,
                                            TextDetectorMixin)
from mmocr.models.textrecog.recognizer import EncodeDecodeRecognizer

from mmdeploy.utils.config_utils import (Backend, Task, get_backend,
                                         get_task_type, load_config)


@DETECTORS.register_module()
class DeployBaseTextDetector(TextDetectorMixin, SingleStageTextDetector):
    """Base Class of Wrapper for TextDetector.

    Args:
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False,
                 *args,
                 **kwargs):
        SingleStageTextDetector.__init__(self, cfg.model.backbone,
                                         cfg.model.neck, cfg.model.bbox_head)
        TextDetectorMixin.__init__(self, show_score)
        self.device_id = device_id
        self.show_score = show_score
        self.cfg = cfg

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def simple_test(self,
                    img: torch.Tensor,
                    img_metas: Sequence[dict],
                    rescale: bool = False,
                    *args,
                    **kwargs):
        """Run forward test.

        Args:
            img (torch.Tensor): Input image tensor.
            img_metas (Sequence[dict]): A list of meta info for image(s).

        Returns:
            list: A list of predictions.
        """
        pred = self.forward_of_backend(img, img_metas, *args, **kwargs)
        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(
                    *(pred[i].unsqueeze(0)), [img_metas[i]], rescale=rescale)
                for i in range(len(img_metas))
            ]

        else:
            boundaries = [
                self.bbox_head.get_boundary(*pred, img_metas, rescale=rescale)
            ]
        return boundaries


@DETECTORS.register_module()
class DeployBaseRecognizer(EncodeDecodeRecognizer):
    """Base Class of Wrapper for TextRecognizer.

    Args:
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False,
                 *args,
                 **kwargs):
        super(DeployBaseRecognizer,
              self).__init__(None, cfg.model.backbone, cfg.model.encoder,
                             cfg.model.decoder, cfg.model.loss,
                             cfg.model.label_convertor, None, None, 40, None)
        self.device_id = device_id
        self.show_score = show_score
        self.cfg = cfg

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward(self, img: Union[torch.Tensor, Sequence[torch.Tensor]],
                img_metas: Sequence[dict], *args, **kwargs):
        """Run forward.

        Args:
            imgs (torch.Tensor | Sequence[torch.Tensor]): Image input tensor.
            img_metas (Sequence[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """

        if isinstance(img, list):
            for idx, each_img in enumerate(img):
                if each_img.dim() == 3:
                    img[idx] = each_img.unsqueeze(0)
            img = img[0]  # avoid aug_test
            img_metas = img_metas[0]
        else:
            if len(img_metas) == 1 and isinstance(img_metas[0], list):
                img_metas = img_metas[0]

        return self.simple_test(img, img_metas, **kwargs)

    def simple_test(self, img: torch.Tensor, img_metas: Sequence[dict], *args,
                    **kwargs):
        """Run forward test.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (Sequence[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        pred = self.forward_of_backend(img, img_metas, *args, **kwargs)
        label_indexes, label_scores = self.label_convertor.tensor2idx(
            pred, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results


class ONNXRuntimeDetector(DeployBaseTextDetector):
    """Wrapper for TextDetector with ONNX Runtime.

    Args:
        model_file (str): The path of input model file.
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 model_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False,
                 *args,
                 **kwargs):
        super(ONNXRuntimeDetector, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_of_backend(self, img: torch.Tensor, img_metas: Iterable, *args,
                           **kwargs):
        """Implement forward test with a backend.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]]): List of image information.
        Returns:
            np.ndarray: Prediction of input model.
        """
        onnx_pred = self.model({'input': img})
        onnx_pred = torch.from_numpy(onnx_pred[0])
        return onnx_pred


class ONNXRuntimeRecognizer(DeployBaseRecognizer):
    """Wrapper for TextRecognizer with ONNX Runtime.

    Args:
        model_file (str): The path of input model file.
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 model_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False,
                 *args,
                 **kwargs):
        super(ONNXRuntimeRecognizer, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(model_file, device_id)

    def forward_of_backend(self, img: torch.Tensor, img_metas: Sequence[dict],
                           *args, **kwargs):
        """Implement forward test with a backend.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]]): List of image information.
        Returns:
            np.ndarray: Prediction of input model.
        """
        onnx_pred = self.model({'input': img})
        onnx_pred = torch.from_numpy(onnx_pred[0])
        return onnx_pred


class TensorRTDetector(DeployBaseTextDetector):
    """Wrapper for TextDetector with TensorRT.

    Args:
        model_file (str): The path of input model file.
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 model_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False,
                 *args,
                 **kwargs):
        super(TensorRTDetector, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.tensorrt import TRTWrapper
        model = TRTWrapper(model_file)
        self.model = model

    def forward_of_backend(self, img: torch.Tensor, img_metas: Sequence[dict],
                           *args, **kwargs):
        """Implement forward test with a backend.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]]): List of image information.
        Returns:
            np.ndarray: Prediction of input model.
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_pred = self.model({'input': img})['output']
        return trt_pred


class TensorRTRecognizer(DeployBaseRecognizer):
    """Wrapper for TextRecognizer with TensorRT.

    Args:
        model_file (str): The path of input model file.
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 model_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False,
                 *args,
                 **kwargs):
        super(TensorRTRecognizer, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.tensorrt import TRTWrapper
        model = TRTWrapper(model_file)
        self.model = model

    def forward_of_backend(self, img: torch.Tensor, img_metas: Sequence[dict],
                           *args, **kwargs):
        """Implement forward test with a backend.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]]): List of image information.
        Returns:
            torch.Tensor: Prediction of input model.
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_pred = self.model({'input': img})['output']
        return trt_pred


class NCNNDetector(DeployBaseTextDetector):
    """Wrapper for TextDetector with NCNN.

    Args:
        model_file (Sequence[str]): Paths of input model files.
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 model_file: Sequence[str],
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
        super(NCNNDetector, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.ncnn import NCNNWrapper
        self.model = NCNNWrapper(
            model_file[0], model_file[1], output_names=['output'])

    def forward_of_backend(self, img: torch.Tensor, img_metas: Sequence[dict],
                           *args, **kwargs):
        """Implement forward test with a backend.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]]): List of image information.
        Returns:
            torch.Tensor: Prediction of input model.
        """
        pred = self.model({'input': img})['output']
        return pred


class NCNNRecognizer(DeployBaseRecognizer):
    """Wrapper for TextRecognizer with NCNN.

    Args:
        model_file (Sequence[str]): Paths of input model files.
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 model_file: Sequence[str],
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
        super(NCNNRecognizer, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.ncnn import NCNNWrapper
        self.model = NCNNWrapper(
            model_file[0], model_file[1], output_names=['output'])

    def forward_of_backend(self, img: torch.Tensor, img_metas: Sequence[dict],
                           *args, **kwargs):
        """Implement forward test with a backend.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]]): List of image information.
        Returns:
            torch.Tensor: Prediction of input model.
        """
        pred = self.model({'input': img})['output']
        return pred


class PPLDetector(DeployBaseTextDetector):
    """Wrapper for TextDetector with PPL.

    Args:
        model_file (Sequence[str]): Paths of input model files.
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 model_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False,
                 *args,
                 **kwargs):
        super(PPLDetector, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.ppl import PPLWrapper
        model = PPLWrapper(model_file[0], model_file[1], device_id)
        self.model = model

    def forward_of_backend(self, img: torch.Tensor, img_metas: Sequence[dict],
                           *args, **kwargs):
        """Implement forward test with a backend.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]]): List of image information.
        Returns:
            torch.Tensor: Prediction of input model.
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            ppl_pred = self.model({'input': img})
        ppl_pred = torch.from_numpy(ppl_pred[0])
        return ppl_pred


class PPLRecognizer(DeployBaseRecognizer):
    """Wrapper for TextRecognizer with PPL.

    Args:
        onnx_file (str): Path of input ONNX model file.
        algo_file (str): Path of PPL algorithm file.
        cfg (str | mmcv.ConfigDict): Input model config.
        device_id (int): An integer represents device index.
        show_score (bool): Whether to show scores. Defaults to `False`.
    """

    def __init__(self,
                 model_file: str,
                 algo_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False,
                 *args,
                 **kwargs):
        super(PPLRecognizer, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.ppl import PPLWrapper
        model = PPLWrapper(model_file, algo_file, device_id)
        self.model = model

    def forward_of_backend(self, img: torch.Tensor, img_metas: Sequence[dict],
                           *args, **kwargs):
        """Implement forward test with a backend.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.
            img_metas (Sequence[dict]]): List of image information.
        Returns:
            torch.Tensor: Prediction of input model.
        """
        with torch.cuda.device(self.device_id), torch.no_grad():
            ppl_pred = self.model({'input': img})[0]
        ppl_pred = torch.from_numpy(ppl_pred[0])
        return ppl_pred


def get_classes_from_config(model_cfg: Union[str, mmcv.Config], **kwargs):
    """Get class name from config.

    Args:
        model_cfg (str | mmcv.Config): Input model config file or
            Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
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

    return module.CLASSES


TASK_ONNXRUNTIME_MAP = {
    Task.TEXT_DETECTION: ONNXRuntimeDetector,
    Task.TEXT_RECOGNITION: ONNXRuntimeRecognizer
}

TASK_TENSORRT_MAP = {
    Task.TEXT_DETECTION: TensorRTDetector,
    Task.TEXT_RECOGNITION: TensorRTRecognizer
}

TASK_PPL_MAP = {
    Task.TEXT_DETECTION: PPLDetector,
    Task.TEXT_RECOGNITION: PPLRecognizer
}

TASK_NCNN_MAP = {
    Task.TEXT_DETECTION: NCNNDetector,
    Task.TEXT_RECOGNITION: NCNNRecognizer
}

BACKEND_TASK_MAP = {
    Backend.ONNXRUNTIME: TASK_ONNXRUNTIME_MAP,
    Backend.TENSORRT: TASK_TENSORRT_MAP,
    Backend.PPL: TASK_PPL_MAP,
    Backend.NCNN: TASK_NCNN_MAP
}


def build_ocr_processor(model_files: Sequence[str],
                        model_cfg: Union[str, mmcv.Config],
                        deploy_cfg: Union[str, mmcv.Config], device_id: int,
                        **kwargs):
    """Build text detector or recognizer for a backend.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device_id (int): An integer represents device index.

    Returns:
        nn.Module: An instance of text detector or recognizer.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    task = get_task_type(deploy_cfg)

    assert backend in BACKEND_TASK_MAP, \
        f'Unsupported backend type: {backend.value}'
    assert task in BACKEND_TASK_MAP[backend], \
        f'Unsupported task type: {task.value}'
    backend_task_class = BACKEND_TASK_MAP[backend][task]

    model_files = model_files[0] if len(model_files) == 1 else model_files
    backend_detector = backend_task_class(
        model_file=model_files, cfg=model_cfg, device_id=device_id, **kwargs)

    return backend_detector
