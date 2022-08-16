# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
from mmocr.models.textdet.heads.base_textdet_head import SampleList

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textdet.heads.BaseTextDetHead.predict')
def base_text_det_head__predict(ctx, self, x: torch.Tensor,
                                batch_data_samples: SampleList) -> SampleList:
    """Rewrite `predict` of BaseTextDetHead for default backend.

    Rewrite this function to early return the results to avoid post processing.
    The process is not suitable for exporting to backends and better get
    implemented in SDK.

    Args:
        x (tuple[Tensor]): Multi-level features from the
            upstream network, each is a 4D-tensor.
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        SampleList: Detection results of each image
        after the post process.
    """
    outs = self(x, batch_data_samples)
    # early return to avoid decoding outputs from heads to boundaries.
    if isinstance(outs, Dict):
        return torch.cat([value.unsqueeze(1) for value in outs.values()], 1)
    return outs


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.textdet.heads.DBHead.forward')
def db_head__forward(ctx,
                     self,
                     img: torch.Tensor,
                     data_samples: Optional[SampleList] = None) -> SampleList:
    """Rewrite `predict` of BaseTextDetHead for default backend.

    Rewrite this function to early return the results to avoid post processing.
    The process is not suitable for exporting to backends and better get
    implemented in SDK.

    Args:
        x (tuple[Tensor]): Multi-level features from the
            upstream network, each is a 4D-tensor.
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        SampleList: Detection results of each image
        after the post process.
    """
    prob_map = self.binarize(img)
    thr_map = self.threshold(img)
    binary_map = self._diff_binarize(prob_map, thr_map, k=50)
    outputs = torch.cat((prob_map, thr_map, binary_map), dim=1)
    return outputs
