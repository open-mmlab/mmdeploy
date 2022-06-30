# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from torch import nn

from mmdeploy.core import MODULE_REWRITER
from mmdeploy.utils import is_dynamic_shape


@MODULE_REWRITER.register_rewrite_module(
    'mmdet.models.detectors.single_stage.SingleStageDetector',
    backend='default')
class SingleStageDetector__default(nn.Module):
    """A patch model for SingleStageDetector.

    `forward` of this class would output a DataElement, which can not be export
    to ONNX.
    """

    def __init__(self, module, deploy_cfg, data_samples, **kwargs) -> None:
        super().__init__()
        self._module = module
        self._deploy_cfg = deploy_cfg
        self._data_samples = data_samples

    def forward(self, batch_inputs):
        data_samples = copy.deepcopy(self._data_samples)
        deploy_cfg = self._deploy_cfg

        # get origin input shape as tensor to support onnx dynamic shape
        is_dynamic_flag = is_dynamic_shape(deploy_cfg)
        img_shape = torch._shape_as_tensor(batch_inputs)[2:]
        if not is_dynamic_flag:
            img_shape = [int(val) for val in img_shape]

        # set the metainfo
        # note that we can not use `set_metainfo`, deepcopy would crash the
        # onnx trace.
        for data_sample in data_samples:
            data_sample.set_field(
                name='img_shape', value=img_shape, field_type='metainfo')

        x = self._module.extract_feat(batch_inputs)

        output = self._module.bbox_head.predict(x, data_samples, rescale=False)
        return output
