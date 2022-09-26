# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmdet3d.core.bbox.structures import limit_period

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.core.bbox.coders.fcos3d_bbox_coder.FCOS3DBBoxCoder.decode_yaw')
def decode_yaw(ctx, self, bbox, centers2d, dir_cls, dir_offset, cam2img):
    """Decode yaw angle and change it from local to global.i. Rewrite this func
    to use slice instead of the original operation.
    Args:
        bbox (torch.Tensor): Bounding box predictions in shape
            [N, C] with yaws to be decoded.
        centers2d (torch.Tensor): Projected 3D-center on the image planes
            corresponding to the box predictions.
        dir_cls (torch.Tensor): Predicted direction classes.
        dir_offset (float): Direction offset before dividing all the
            directions into several classes.
        cam2img (torch.Tensor): Camera intrinsic matrix in shape [4, 4].
    Returns:
        torch.Tensor: Bounding boxes with decoded yaws.
    """
    if bbox.shape[0] > 0:
        dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi)
        bbox[..., 6] = \
            dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype)

    bbox[..., 6] = torch.atan2(centers2d[..., 0] - cam2img[0, 2],
                               cam2img[0, 0]) + bbox[..., 6]

    return bbox
