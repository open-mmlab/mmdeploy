import warnings
from typing import Iterable, Union

import mmcv
import torch
from mmdet.models.builder import DETECTORS
from mmocr.models.textdet.detectors import (SingleStageTextDetector,
                                            TextDetectorMixin)
from mmocr.models.textrecog.recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class DeployBaseTextDetector(TextDetectorMixin, SingleStageTextDetector):
    """DeployBaseTextDetector."""

    def __init__(self,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
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
                    rescale: bool = False):
        pred = self.forward_of_backend(img, img_metas, rescale)
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
                 show_score: bool = False):
        EncodeDecodeRecognizer.__init__(self, None, cfg.model.backbone,
                                        cfg.model.encoder, cfg.model.decoder,
                                        cfg.model.loss,
                                        cfg.model.label_convertor, None, None,
                                        40, None)
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
                    rescale: bool = False):
        """Test function.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        pred = self.forward_of_backend(img, img_metas, rescale)
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
                 onnx_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
        super(ONNXRuntimeDetector, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(onnx_file, device_id)

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False):
        onnx_pred = self.model(img)
        onnx_pred = torch.from_numpy(onnx_pred[0])
        return onnx_pred


class ONNXRuntimeRecognizer(DeployBaseRecognizer):
    """The class for evaluating onnx file of recognition."""

    def __init__(self,
                 onnx_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
        super(ONNXRuntimeRecognizer, self).__init__(cfg, device_id, show_score)
        from mmdeploy.apis.onnxruntime import ORTWrapper
        self.model = ORTWrapper(onnx_file, device_id)

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False):
        onnx_pred = self.model(img)
        onnx_pred = torch.from_numpy(onnx_pred[0])
        return onnx_pred


class TensorRTDetector(DeployBaseTextDetector):
    """The class for evaluating TensorRT file of text detection."""

    def __init__(self,
                 trt_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
        super(TensorRTDetector, self).__init__(cfg, device_id, show_score)
        from mmcv.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with TensorRT from source.')
        model = TRTWrapper(
            trt_file, input_names=['input'], output_names=['output'])
        self.model = model

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False):
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_pred = self.model({'input': img})['output']
        return trt_pred


class TensorRTRecognizer(DeployBaseRecognizer):
    """The class for evaluating TensorRT file of recognition."""

    def __init__(self,
                 trt_file: str,
                 cfg: Union[mmcv.Config, mmcv.ConfigDict],
                 device_id: int,
                 show_score: bool = False):
        super(TensorRTRecognizer, self).__init__(cfg, device_id, show_score)
        from mmcv.tensorrt import TRTWrapper, load_tensorrt_plugin
        try:
            load_tensorrt_plugin()
        except (ImportError, ModuleNotFoundError):
            warnings.warn('If input model has custom op from mmcv, \
                you may have to build mmcv with TensorRT from source.')
        model = TRTWrapper(
            trt_file, input_names=['input'], output_names=['output'])
        self.model = model

    def forward_of_backend(self,
                           img: torch.Tensor,
                           img_metas: Iterable,
                           rescale: bool = False):
        with torch.cuda.device(self.device_id), torch.no_grad():
            trt_pred = self.model({'input': img})['output']
        return trt_pred
