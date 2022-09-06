# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER
from typing import List, Union
from mmdet3d.structures.det3d_data_sample import (OptSampleList)

@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.base.Base3DDetector.forward')

def base3ddetector__forward_test(ctx,
        self,
        inputs: Union[dict, List[dict]],
        data_samples: OptSampleList = None,
        mode: str = 'tensor',
        **kwargs):
    """Rewrite this function to run simple_test directly."""
    return self._forward(inputs, data_samples, **kwargs)
