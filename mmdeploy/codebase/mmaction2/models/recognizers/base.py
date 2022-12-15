# Copyright (c) OpenMMLab. All rights reserved.

from mmaction.utils import OptSampleList
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmaction.models.recognizers.BaseRecognizer.forward')
def base_recognizer__forward(self,
                             inputs: Tensor,
                             data_samples: OptSampleList = None,
                             mode: str = 'tensor',
                             **kwargs):
    """Rewrite `forward` of Recognizer2D for default backend.

    Args:
        inputs (torch.Tensor): The input tensor with shape
            (N, C, ...) in general.
        data_samples (List[``ActionDataSample``], optional): The
            annotation data of every samples. Defaults to None.
        mode (str): Return what kind of value. Defaults to ``tensor``.

    Returns:
        return a list of `ActionDataSample`
    """

    assert mode == 'predict'

    feats, predict_kwargs = self.extract_feat(inputs, test_mode=True)
    cls_scores = self.cls_head(feats, **predict_kwargs)
    num_segs = cls_scores.shape[0] // len(data_samples)
    cls_scores = self.cls_head.average_clip(cls_scores, num_segs=num_segs)

    return cls_scores
