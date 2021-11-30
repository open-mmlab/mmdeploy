# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet.deploy import clip_bboxes


def distance2bbox(points, distance, max_shape=None):
    """Rewrite `mmdet.core.bbox.transforms.distance2bbox`

    Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        # clip bboxes with dynamic `min` and `max`
        x1, y1, x2, y2 = clip_bboxes(x1, y1, x2, y2, max_shape)
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return bboxes

    return bboxes
