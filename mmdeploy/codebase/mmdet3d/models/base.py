# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.Base3DDetector.forward'  # noqa: E501
)
def basedetector__forward(self,
                          voxels: torch.Tensor,
                          num_points: torch.Tensor,
                          coors: torch.Tensor,
                          data_samples=None,
                          **kwargs) -> Tuple[List[torch.Tensor]]:
    """Extract features of images."""

    batch_inputs_dict = {
        'voxels': {
            'voxels': voxels,
            'num_points': num_points,
            'coors': coors
        }
    }
    return self._forward(batch_inputs_dict, data_samples, **kwargs)
