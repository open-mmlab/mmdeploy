# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.utils import Registry
from mmseg.datasets import DATASETS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)


def __build_backend_model(cls_name: str, registry: Registry, *args, **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_segmentors', build_func=__build_backend_model)


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of segmentation.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        class_names (Sequence[str]): A list of string specifying class names.
        palette (np.ndarray): The palette of segmentation map.
        deploy_cfg (str | mmcv.Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 class_names: Sequence[str],
                 palette: np.ndarray,
                 deploy_cfg: Union[str, mmcv.Config] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(deploy_cfg=deploy_cfg)
        self.CLASSES = class_names
        self.PALETTE = palette
        self.deploy_cfg = deploy_cfg
        self._init_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            **kwargs)

    def _init_wrapper(self, backend, backend_files, device, **kwargs):
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg,
            **kwargs)

    def forward(self, img: Sequence[torch.Tensor],
                img_metas: Sequence[Sequence[dict]], *args, **kwargs):
        """Run forward inference.

        Args:
            img (Sequence[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        input_img = img[0].contiguous()
        outputs = self.forward_test(input_img, img_metas, *args, **kwargs)
        seg_pred = outputs[0]
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

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs) -> \
            List[np.ndarray]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            List[np.ndarray]: A list of segmentation map.
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [out.detach().cpu().numpy() for out in outputs]
        return outputs

    def show_result(self,
                    img: np.ndarray,
                    result: list,
                    win_name: str = '',
                    palette: Optional[np.ndarray] = None,
                    show: bool = True,
                    opacity: float = 0.5,
                    out_file: str = None):
        """Show predictions of segmentation.
        Args:
            img: (np.ndarray): Input image to draw predictions.
            result (list): A list of predictions.
            win_name (str): The name of visualization window. Default is ''.
            palette (np.ndarray): The palette of segmentation map.
            show (bool): Whether to show plotted image in windows. Defaults to
                `True`.
            opacity: (float): Opacity of painted segmentation map.
                    Defaults to `0.5`.
            out_file (str): Output image file to save drawn predictions.

        Returns:
            np.ndarray: Drawn image, only if not `show` or `out_file`.
        """
        palette = self.PALETTE if palette is None else palette
        return BaseSegmentor.show_result(
            self,
            img,
            result,
            palette=palette,
            opacity=opacity,
            show=show,
            win_name=win_name,
            out_file=out_file)


@__BACKEND_MODEL.register_module('rknn')
class RKNNModel(End2EndModel):
    """SDK inference class, converts RKNN output to mmseg format."""

    def forward_test(self, imgs: torch.Tensor, *args, **kwargs) -> \
            List[np.ndarray]:
        """The interface for forward test.

        Args:
            imgs (torch.Tensor): Input image(s) in [N x C x H x W] format.

        Returns:
            List[np.ndarray]: A list of segmentation map.
        """
        outputs = self.wrapper({self.input_name: imgs})
        outputs = [output.argmax(dim=1, keepdim=True) for output in outputs]
        outputs = [out.detach().cpu().numpy() for out in outputs]
        return outputs


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmseg format."""

    def forward(self, img: Sequence[torch.Tensor],
                img_metas: Sequence[Sequence[dict]], *args, **kwargs):
        """Run forward inference.

        Args:
            img (Sequence[torch.Tensor]): A list contains input image(s)
                in [N x C x H x W] format.
            img_metas (Sequence[Sequence[dict]]): A list of meta info for
                image(s).
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        masks = self.wrapper.invoke(img[0].contiguous().detach().cpu().numpy())
        return masks


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

    if 'val' in data_cfg:
        module = module_dict[data_cfg.val.type]
    elif 'test' in data_cfg:
        module = module_dict[data_cfg.test.type]
    elif 'train' in data_cfg:
        module = module_dict[data_cfg.train.type]
    else:
        raise RuntimeError(f'No dataset config found in: {model_cfg}')

    return module.CLASSES, module.PALETTE


def build_segmentation_model(model_files: Sequence[str],
                             model_cfg: Union[str, mmcv.Config],
                             deploy_cfg: Union[str, mmcv.Config], device: str,
                             **kwargs):
    """Build object segmentation model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.

    Returns:
        BaseBackendModel: Segmentor for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')
    class_names, palette = get_classes_palette_from_config(model_cfg)

    backend_segmentor = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        device=device,
        class_names=class_names,
        palette=palette,
        deploy_cfg=deploy_cfg,
        **kwargs)

    return backend_segmentor
