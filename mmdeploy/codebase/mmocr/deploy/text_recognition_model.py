# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.utils import Registry
from mmocr.models.builder import build_convertor
from mmocr.models.textrecog import BaseRecognizer

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_text_recognizer', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of text detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (str | mmcv.Config): Deployment config file or loaded Config
            object.
        model_cfg (str | mmcv.Config): Model config file or loaded Config
            object.
    """

    def __init__(
        self,
        backend: Backend,
        backend_files: Sequence[str],
        device: str,
        deploy_cfg: Union[str, mmcv.Config] = None,
        model_cfg: Union[str, mmcv.Config] = None,
    ):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)
        model_cfg, deploy_cfg = load_config(model_cfg, deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.show_score = False
        label_convertor = model_cfg.model.label_convertor
        assert label_convertor is not None, 'model_cfg contains no label '
        'convertor'
        max_seq_len = 40  # default value in EncodeDecodeRecognizer of mmocr
        label_convertor.update(max_seq_len=max_seq_len)
        self.label_convertor = build_convertor(label_convertor)
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        """Initialize the wrapper of backends.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. .onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string represents device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    def forward(self, img: Sequence[torch.Tensor],
                img_metas: Sequence[Sequence[dict]], *args, **kwargs):
        """Run forward inference.

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

        return self.forward_test(img, img_metas, **kwargs)

    def forward_test(self, imgs: torch.Tensor,
                     img_metas: Sequence[Sequence[dict]], *args, **kwargs) -> \
            List[np.ndarray]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (Sequence[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        pred = self.wrapper({self.input_name: imgs})['output']
        label_indexes, label_scores = self.label_convertor.tensor2idx(
            pred, img_metas)
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results

    def show_result(self,
                    img: np.ndarray,
                    result: list,
                    win_name: str,
                    show: bool = True,
                    score_thr: float = 0.3,
                    out_file: str = None):
        """Show predictions of segmentation.
        Args:
            img: (np.ndarray): Input image to draw predictions.
            result (list): A list of predictions.
            win_name (str): The name of visualization window.
            show (bool): Whether to show plotted image in windows. Defaults to
                `True`.
            score_thr: (float): The thresh of score. Defaults to `0.3`.
            out_file (str): Output image file to save drawn predictions.

        Returns:
            np.ndarray: Drawn image, only if not `show` or `out_file`.
        """
        return BaseRecognizer.show_result(
            self,
            img,
            result,
            score_thr=score_thr,
            show=show,
            win_name=win_name,
            out_file=out_file)


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmocr format."""

    def forward(self, img: Sequence[torch.Tensor],
                img_metas: Sequence[Sequence[dict]], *args, **kwargs):
        """Run forward inference.

        Args:
            imgs (torch.Tensor | Sequence[torch.Tensor]): Image input tensor.
            img_metas (Sequence[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        results = self.wrapper.invoke(
            [img[0].contiguous().detach().cpu().numpy()])
        results = [dict(text=text, score=score) for text, score in results]
        return results


def build_text_recognition_model(model_files: Sequence[str],
                                 model_cfg: Union[str, mmcv.Config],
                                 deploy_cfg: Union[str, mmcv.Config],
                                 device: str, **kwargs):
    """Build text recognition model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Text recognizer for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_text_recognizer = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        device=device,
        deploy_cfg=deploy_cfg,
        model_cfg=model_cfg,
        **kwargs)

    return backend_text_recognizer
