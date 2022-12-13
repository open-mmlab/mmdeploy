# Copyright (c) OpenMMLab. All rights reserved.in

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import get_root_logger


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.decode_heads.point_head.PointHead.get_points_test',
    backend='tensorrt')
def point_head__get_points_test__tensorrt(self, seg_logits, uncertainty_func,
                                          cfg):
    """Sample points for testing.

    1. set `num_points` no greater than TENSORRT_MAX_TOPK for tensorrt backend

    Args:
        seg_logits (Tensor): A tensor of shape (batch_size, num_classes,
            height, width) for class-specific or class-agnostic prediction.
        uncertainty_func (func): uncertainty calculation function.
        cfg (dict): Testing config of point head.
    Returns:
        point_indices (Tensor): A tensor of shape (batch_size, num_points)
            that contains indices from [0, height x width) of the most
            uncertain points.
        point_coords (Tensor): A tensor of shape (batch_size, num_points,
            2) that contains [0, 1] x [0, 1] normalized coordinates of the
            most uncertain points from the ``height x width`` grid .
    """
    ctx = FUNCTION_REWRITER.get_context()
    from mmdeploy.utils.constants import TENSORRT_MAX_TOPK

    if cfg.subdivision_num_points > TENSORRT_MAX_TOPK:
        logger = get_root_logger()
        logger.warning(f'cfg.subdivision_num_points would be changed from '
                       f'{cfg.subdivision_num_points} to {TENSORRT_MAX_TOPK} '
                       f'due to restriction in TensorRT TopK layer ')
        cfg.subdivision_num_points = TENSORRT_MAX_TOPK
    return ctx.origin_func(self, seg_logits, uncertainty_func, cfg)
