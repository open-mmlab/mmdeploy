# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.Base3DDetector.forward'  # noqa: E501
)
def basedetector__forward(self,
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
    return self._forward(batch_inputs_dict, data_samples, **kwargs)
