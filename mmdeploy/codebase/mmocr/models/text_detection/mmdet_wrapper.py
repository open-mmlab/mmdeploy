# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
from mmdet.structures import DetDataSample
from mmdet.structures import SampleList as MMDET_SampleList
from mmocr.structures import TextDetDataSample
from mmocr.utils.typing_utils import DetSampleList

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textdet.detectors.MMDetWrapper.forward')
def mmdet_wrapper__forward(self,
                           inputs: torch.Tensor,
                           data_samples: Optional[Union[
                               DetSampleList, MMDET_SampleList]] = None,
                           mode: str = 'tensor',
                           **kwargs) -> Sequence[TextDetDataSample]:
    """The unified entry for a forward process in both training and test.

    The method works in three modes: "tensor", "predict" and "loss":

    - "tensor": Forward the whole network and return tensor or tuple of
    tensor without any post-processing, same as a common nn.Module.
    - "predict": Forward and return the predictions, which are fully
    processed to a list of :obj:`DetDataSample`.
    - "loss": Forward and return a dict of losses according to the given
    inputs and data samples.

    Note that this method doesn't handle either back propagation or
    parameter update, which are supposed to be done in :meth:`train_step`.

    Args:
        inputs (torch.Tensor): The input tensor with shape
            (N, C, ...) in general.
        data_samples (list[:obj:`DetDataSample`] or
            list[:obj:`TextDetDataSample`]): The annotation data of every
            sample. When in "predict" mode, it should be a list of
            :obj:`TextDetDataSample`. Otherwise they are
            :obj:`DetDataSample`s. Defaults to None.
        mode (str): Running mode. Defaults to 'tensor'.

    Returns:
        results (Sequence(torch.Tensor)): Output of MMDet models.
    """
    if mode == 'predict':
        ocr_data_samples = data_samples
        data_samples = []
        for i in range(len(ocr_data_samples)):
            data_samples.append(
                DetDataSample(metainfo=ocr_data_samples[i].metainfo))

    results = self.wrapped_model.forward(inputs, data_samples, mode, **kwargs)
    return results
