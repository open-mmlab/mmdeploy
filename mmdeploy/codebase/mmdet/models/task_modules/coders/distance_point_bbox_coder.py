# Copyright (c) OpenMMLab. All rights reserved.
import mmdet.structures.bbox.transforms

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.task_modules.coders.distance_point_bbox_coder'
    '.DistancePointBBoxCoder.decode',
    backend='default')
def distancepointbboxcoder__decode(self, points, pred_bboxes, max_shape=None):
    """Rewrite `mmdet.models.task_modules.coders.distance_point_bbox_coder. \
    DistancePointBBoxCoder.decode`

    Decode distance prediction to bounding box.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (DistancePointBBoxCoder): The instance of the class
            DistancePointBBoxCoder.
        points (Tensor): Shape (B, N, 2) or (N, 2).
        pred_bboxes (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4)
            or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]],
            and the length of max_shape should also be B.
            Default None.
    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """
    assert points.size(0) == pred_bboxes.size(0)
    assert points.size(-1) == 2
    assert pred_bboxes.size(-1) == 4
    if self.clip_border is False:
        max_shape = None
    # Rewrite add mmdet.core.bbox.transforms to find correct
    # rewriter, or you will not find correct rewriter.
    return mmdet.structures.bbox.transforms.distance2bbox(
        points, pred_bboxes, max_shape)
