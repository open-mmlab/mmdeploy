# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.single_stage.SingleStage3DDetector.predict'  # noqa: E501
)
def singlestagedetector__predict(ctx,
                                 self,
                                 inputs: list,
                                 data_samples=None,
                                 **kwargs) -> Tuple[List[torch.Tensor]]:
    """Extract features of images."""

    batch_inputs_dict = {
        'voxels': {
            'voxels': inputs[0],
            'num_points': inputs[1],
            'coors': inputs[2]
        }
    }

    x = self.extract_feat(batch_inputs_dict)
    results = self.bbox_head.forward(x)
    return results
