from typing import Iterable, Union

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
    """DeployBaseTextDetector."""

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
                    img_metas: Iterable,
                    rescale: bool = False,
                    *args,
                    **kwargs):
        pred = self.forward_of_backend(img, img_metas, rescale, *args,
                                       **kwargs)
        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(pred[i].unsqueeze(0)),
                                            [img_metas[i]], rescale)
                for i in range(len(img_metas))
            ]

        else:
            boundaries = [
                self.bbox_head.get_boundary(*pred, img_metas, rescale)
            ]
        return boundaries


@DETECTORS.register_module()
class DeployBaseRecognizer(EncodeDecodeRecognizer):

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

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note that img and img_meta are single-nested (i.e. tensor and
        list[dict]).
        """

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)

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

    def simple_test(self,
                    img: torch.Tensor,
                    img_metas: Iterable,
                    rescale: bool = False,
                    *args,
                    **kwargs):
        """Test function.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        pred = self.forward_of_backend(img, img_metas, rescale, *args,
                                       **kwargs)
        label_indexes, label_scores = self.label_convertor.tensor2idx(
            pred, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results


class ONNXRuntimeDetector(DeployBaseTextDetector):
    """The class for evaluating onnx file of detection."""

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

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False,
                           *args,
                           **kwargs):
        onnx_pred = self.model({'input': img})
        onnx_pred = torch.from_numpy(onnx_pred[0])
        return onnx_pred


class ONNXRuntimeRecognizer(DeployBaseRecognizer):
    """The class for evaluating onnx file of recognition."""

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

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False,
                           *args,
                           **kwargs):
        onnx_pred = self.model({'input': img})
        onnx_pred = torch.from_numpy(onnx_pred[0])
        return onnx_pred


class TensorRTDetector(DeployBaseTextDetector):
    """The class for evaluating TensorRT file of text detection."""

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

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False,
                           *args,
                           **kwargs):
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_pred = self.model({'input': img})['output']
        return trt_pred


class TensorRTRecognizer(DeployBaseRecognizer):
    """The class for evaluating TensorRT file of recognition."""

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

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False,
                           *args,
                           **kwargs):
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_pred = self.model({'input': img})['output']
        return trt_pred


class NCNNDetector(DeployBaseTextDetector):
    """The class for evaluating NCNN file of text detection."""

    def __init__(self,
                 model_file: Iterable[str],
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
        super(NCNNDetector, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.ncnn import NCNNWrapper
        self.model = NCNNWrapper(
            model_file[0], model_file[1], output_names=['output'])

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False):
        pred = self.model({'input': img})['output']
        return pred


class NCNNRecognizer(DeployBaseRecognizer):
    """The class for evaluating NCNN file of recognition."""

    def __init__(self,
                 model_file: Iterable[str],
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
        super(NCNNRecognizer, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.ncnn import NCNNWrapper
        self.model = NCNNWrapper(
            model_file[0], model_file[1], output_names=['output'])

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False):
        pred = self.model({'input': img})['output']
        return pred


def get_classes_from_config(model_cfg: Union[str, mmcv.Config], **kwargs):
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

TASK_NCNN_MAP = {
    Task.TEXT_DETECTION: NCNNDetector,
    Task.TEXT_RECOGNITION: NCNNRecognizer
}

BACKEND_TASK_MAP = {
    Backend.ONNXRUNTIME: TASK_ONNXRUNTIME_MAP,
    Backend.TENSORRT: TASK_TENSORRT_MAP,
    Backend.NCNN: TASK_NCNN_MAP
}


def build_ocr_processor(model_files, model_cfg, deploy_cfg, device_id,
                        **kwargs):
    # load cfg if necessary
    deploy_cfg = load_config(deploy_cfg)[0]
    model_cfg = load_config(model_cfg)[0]

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
