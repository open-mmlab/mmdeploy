# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from mmocr.structures import TextDetDataSample

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textdet.SingleStageTextDetector.forward')
def single_stage_text_detector__forward(
        self,
        batch_inputs: torch.Tensor,
        data_samples: TextDetDataSample = None,
        **kwargs) -> Sequence[TextDetDataSample]:
    """Predict results from a batch of inputs and data samples with post-
    processing.

    Args:
        batch_inputs (torch.Tensor): Images of shape (N, C, H, W).
        data_samples (list[TextDetDataSample]): A list of N
            datasamples, containing meta information and gold annotations
            for each of the images.

    Returns:
        list[TextDetDataSample]: A list of N datasamples of prediction
        results.  Each DetDataSample usually contain
        'pred_instances'. And the ``pred_instances`` usually
        contains following keys.

            - scores (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
                the last dimension 4 arrange as (x1, y1, x2, y2).
            - polygons (list[np.ndarray]): The length is num_instances.
                Each element represents the polygon of the
                instance, in (xn, yn) order.
    """
    x = self.extract_feat(batch_inputs)
    return self.det_head.predict(x, data_samples)
