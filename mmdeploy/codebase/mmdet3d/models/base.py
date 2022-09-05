# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.base.Base3DDetector.forward')
def base3ddetector__forward_test(ctx,
                                 self,
                                 voxels,
                                 num_points,
                                 coors,
                                 img_metas=None,
                                 img=None,
                                 rescale=False,
                                 **kwargs):
    """Rewrite this function to run simple_test directly."""
    return self.simple_test(voxels, num_points, coors, img_metas, img)
