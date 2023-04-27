# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import torch
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement, PixelData
from torch import nn

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_root_logger, load_config)

__BACKEND_MODEL = Registry('backend_segmentors')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of segmentation.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (str | mmengine.Config): Deployment config file or loaded
            Config object.
        data_preprocessor (dict | nn.Module | None): Input data pre-
            processor. Default is ``None``.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 deploy_cfg: Union[str, Config] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.device = device
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

    def forward(self,
                inputs: torch.Tensor,
                data_samples: List[BaseDataElement],
                mode: str = 'predict',
                **kwargs):
        """Run forward inference.

        Args:
            inputs (torch.Tensor): Input image tensor
                in [N x C x H x W] format.
            data_samples (list[:obj:`SegDataSample`]): The seg data
                samples. It usually includes information such as
                `metainfo` and `gt_sem_seg`. Default to None.
            mode (str): forward mode, only support 'predict'.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        if inputs.device != torch.device(self.device):
            get_root_logger().warning(f'expect input device {self.device}'
                                      f' but get {inputs.device}.')
        inputs = inputs.to(self.device)
        batch_outputs = self.wrapper({self.input_name:
                                      inputs})[self.output_names[0]]
        return self.pack_result(batch_outputs, data_samples)

    def pack_result(self, batch_outputs: torch.Tensor,
                    data_samples: List[BaseDataElement]):
        """Pack segmentation result to data samples.
        Args:
            batch_outputs (Tensor): Batched segmentation output
                tensor.
            data_samples (list[:obj:`SegDataSample`]): The seg data
                samples. It usually includes information such as
                `metainfo` and `gt_sem_seg`. Default to None.

        Returns:
            list[:obj:`SegDataSample`]: The updated seg data samples.
        """

        predictions = []
        for seg_pred, data_sample in zip(batch_outputs, data_samples):
            # resize seg_pred to original image shape
            metainfo = data_sample.metainfo
            if get_codebase_config(self.deploy_cfg).get('with_argmax',
                                                        True) is False:
                seg_pred = seg_pred.argmax(dim=0, keepdim=True)
            if metainfo['ori_shape'] != metainfo['img_shape']:
                from mmseg.models.utils import resize
                ori_type = seg_pred.dtype
                seg_pred = resize(
                    seg_pred.unsqueeze(0).to(torch.float32),
                    size=metainfo['ori_shape'],
                    mode='nearest').squeeze(0).to(ori_type)
            data_sample.set_data(
                dict(pred_sem_seg=PixelData(**dict(data=seg_pred))))
            predictions.append(data_sample)

        return predictions


@__BACKEND_MODEL.register_module('vacc_seg')
class VACCModel(End2EndModel):
    """SDK inference class, converts VACC output to mmseg format."""

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict'):
        """Run forward inference.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (list[:obj:`SegDataSample`]): The seg data
                samples. It usually includes information such as
                `metainfo` and `gt_sem_seg`. Default to None.

        Returns:
            list: A list contains predictions.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        if inputs.device != torch.device(self.device):
            get_root_logger().warning(f'expect input device {self.device}'
                                      f' but get {inputs.device}.')
        inputs = inputs.to(self.device)
        batch_outputs = self.wrapper({self.input_name: inputs})
        batch_outputs = [
            output.argmax(dim=1, keepdim=True)
            for output in batch_outputs.values()
        ]
        return self.pack_result(batch_outputs[0], data_samples)


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmseg format."""

    def __init__(self, *args, **kwargs):
        kwargs['data_preprocessor'] = None
        super(SDKEnd2EndModel, self).__init__(*args, **kwargs)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict'):
        """Run forward inference.

        Args:
            inputs (Sequence[torch.Tensor]): A list contains input
                image(s) in [C x H x W] format.
            data_samples (list[:obj:`SegDataSample`]): The seg data
                samples. It usually includes information such as
                `metainfo` and `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to
                'predict'.

        Returns:
            list: A list contains predictions.
        """
        if isinstance(inputs, list):
            inputs = inputs[0]
        # inputs are c,h,w, sdk requested h,w,c
        inputs = inputs.permute(1, 2, 0)
        outputs = self.wrapper.invoke(
            inputs.contiguous().detach().cpu().numpy())
        batch_outputs = torch.from_numpy(outputs).to(torch.int64).to(
            self.device)
        batch_outputs = batch_outputs.unsqueeze(0).unsqueeze(0)
        return self.pack_result(batch_outputs, data_samples)


def build_segmentation_model(
        model_files: Sequence[str],
        model_cfg: Union[str, Config],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build object segmentation model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmengine.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmengine.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.
        data_preprocessor (BaseDataPreprocessor | Config): The data
            preprocessor of the model.

    Returns:
        BaseBackendModel: Segmentor for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_segmentor = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_segmentor
