# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from mmengine.structures import BaseDataElement
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmcls.models.classifiers.ImageClassifier.forward', backend='default')
@FUNCTION_REWRITER.register_rewriter(
    'mmcls.models.classifiers.BaseClassifier.forward', backend='default')
def base_classifier__forward(
        ctx,
        self,
        batch_inputs: Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
        mode: str = 'predict'):
    """Rewrite `forward` of BaseClassifier for default backend.

    Args:
        batch_inputs (torch.Tensor): The input tensor with shape
            (N, C, ...) in general.
        data_samples (List[BaseDataElement], optional): The annotation
            data of every samples. It's required if ``mode="loss"``.
            Defaults to None.
        mode (str): Return what kind of value. Defaults to 'predict'.

    Returns:
        return a list of :obj:`mmengine.BaseDataElement`.
    """
    output = self.extract_feat(batch_inputs)
    if self.head is not None:
        output = self.head(output)
    return output
