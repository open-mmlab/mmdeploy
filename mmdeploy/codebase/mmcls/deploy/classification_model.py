# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine import Config
from mmengine.model import BaseDataPreprocessor
from mmengine.registry import Registry
from mmengine.structures import BaseDataElement
from torch import nn

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_root_logger, load_config)

__BACKEND_MODEL = Registry('backend_classifiers')


@__BACKEND_MODEL.register_module('end2end')
class End2EndModel(BaseBackendModel):
    """End to end model for inference of classification.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files(e.g.
            '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string represents device type.
        deploy_cfg (str | Config): Deployment config file or loaded Config
            object.
        data_preprocessor (BaseDataPreprocessor): The data preprocessor
                of the model. Default to `None`.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: Union[str, Config] = None,
                 deploy_cfg: Union[str, Config] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 **kwargs):
        super(End2EndModel, self).__init__(
            deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self._init_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            **kwargs)
        self.model_cfg = model_cfg
        self.head = None
        if model_cfg is not None:
            self.head = self._get_head()
        self.device = device

    def _get_head(self):
        from mmcls.models import build_head
        head_config = self.model_cfg['model']['head']
        head = build_head(head_config)
        return head

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str, **kwargs):
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
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict') -> Any:
        """Run forward inference.

        Args:
            inputs (torch.Tensor): The input tensors
            data_samples (List[BaseDataElement], optional): The data samples.
                Defaults to None.
            mode (str, optional): forward mode, only support `predict`.

        Returns:
            Any: Model output.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        if inputs.device != torch.device(self.device):
            get_root_logger().warning(f'expect input device {self.device}'
                                      f' but get {inputs.device}.')
        inputs = inputs.to(self.device)
        cls_score = self.wrapper({self.input_name:
                                  inputs})[self.output_names[0]]

        from mmcls.models.heads import MultiLabelClsHead
        from mmcls.structures import ClsDataSample
        pred_scores = cls_score

        if self.head is None or not isinstance(self.head, MultiLabelClsHead):
            pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

            if data_samples is not None:
                for data_sample, score, label in zip(data_samples, pred_scores,
                                                     pred_labels):
                    data_sample.set_pred_score(score).set_pred_label(label)
            else:
                data_samples = []
                for score, label in zip(pred_scores, pred_labels):
                    data_samples.append(ClsDataSample().set_pred_score(
                        score).set_pred_label(label))
        else:
            if data_samples is None:
                data_samples = [
                    ClsDataSample() for _ in range(cls_score.size(0))
                ]

            for data_sample, score in zip(data_samples, pred_scores):
                if self.head.thr is not None:
                    # a label is predicted positive if larger than thr
                    label = torch.where(score >= self.head.thr)[0]
                else:
                    # top-k labels will be predicted positive for any example
                    _, label = score.topk(self.head.topk)
                data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples


@__BACKEND_MODEL.register_module('sdk')
class SDKEnd2EndModel(End2EndModel):
    """SDK inference class, converts SDK output to mmcls format."""

    def __init__(self, *arg, **kwargs):
        kwargs['data_preprocessor'] = None
        super().__init__(*arg, **kwargs)

    def forward(self,
                inputs: Sequence[torch.Tensor],
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict',
                *args,
                **kwargs) -> list:
        """Run forward inference.

        Args:
            inputs (torch.Tensor): The input tensors
            data_samples (List[BaseDataElement], optional): The data samples.
                Defaults to None.
            mode (str, optional): forward mode, only support `predict`.
            *args: Other arguments.
            **kwargs: Other key-pair arguments.

        Returns:
            list: A list contains predictions.
        """
        cls_score = []
        for input in inputs:
            pred = self.wrapper.invoke(
                input.permute(1, 2, 0).contiguous().detach().cpu().numpy())
            pred = np.array(pred, dtype=np.float32)
            pred = pred[np.argsort(pred[:, 0])][np.newaxis, :, 1]
            cls_score.append(torch.from_numpy(pred).to(self.device))

        cls_score = torch.cat(cls_score, 0)
        from mmcls.models.heads.cls_head import ClsHead
        predict = ClsHead._get_predictions(
            None, cls_score, data_samples=data_samples)
        return predict


@__BACKEND_MODEL.register_module('rknn')
class RKNNEnd2EndModel(End2EndModel):
    """RKNN inference class, converts RKNN output to mmcls format."""

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'predict') -> Any:
        """Run forward inference.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_label`, `pred_label`, `scores` and `logits`.
            mode (str, optional): forward mode, only support `predict`.

        Returns:
            Any: Model output.
        """
        assert mode == 'predict', \
            'Backend model only support mode==predict,' f' but get {mode}'
        if inputs.device != torch.device(self.device):
            get_root_logger().warning(f'expect input device {self.device}'
                                      f' but get {inputs.device}.')
        inputs = inputs.to(self.device)
        cls_score = self.wrapper({self.input_name: inputs})[0]

        from mmcls.models.heads.cls_head import ClsHead
        predict = ClsHead._get_predictions(
            None, cls_score, data_samples=data_samples)

        return predict


def build_classification_model(
        model_files: Sequence[str],
        model_cfg: Union[str, Config],
        deploy_cfg: Union[str, Config],
        device: str,
        data_preprocessor: Optional[Union[Config,
                                          BaseDataPreprocessor]] = None,
        **kwargs):
    """Build classification model for different backend.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | Config): Input model config file or Config
            object.
        deploy_cfg (str | Config): Input deployment config file or
            Config object.
        device (str):  Device to input model.
        data_preprocessor (BaseDataPreprocessor): The data preprocessor
                of the model. Default to `None`.
        **kwargs: Other key-pair arguments.

    Returns:
        BaseBackendModel: Classifier for a configured backend.
    """
    # load cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_classifier = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_classifier
