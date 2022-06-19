# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from mmdeploy.core import MODULE_REWRITER


@MODULE_REWRITER.register_rewrite_module(
    'mmcls.models.classifiers.ImageClassifier', backend='default')
class ImageClassifier__default(nn.Module):
    """A patch model for Image classifier.

    `forward` of ImageClassifier would output a DataElement, which can not be
    export to ONNX.
    """

    def __init__(self, module, deploy_cfg, data_samples, **kwargs) -> None:
        super().__init__()
        self._module = module
        self._deploy_cfg = deploy_cfg
        self._data_samples = data_samples

    def forward(self, batch_inputs):
        feats = self._module.extract_feat(batch_inputs)
        output = self._module.head(feats)
        return output
