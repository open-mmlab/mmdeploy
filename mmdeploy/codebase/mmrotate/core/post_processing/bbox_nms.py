# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmdeploy.mmcv.ops import ONNXNMSRotatedOp


def multiclass_nms_rotated(multi_bboxes: Tensor,
                           multi_scores: Tensor,
                           iou_threshold: float = 0.1,
                           score_threshold: float = 0.05,
                           keep_top_k: int = -1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_threshold (float): bbox threshold, bboxes with scores lower than
            it will not be considered.
        keep_top_k (int, optional): if there are more than keep_top_k bboxes
            after NMS, only top keep_top_k will be kept. Default to -1.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category

    bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_threshold

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        return dets, labels

    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes.clone()
    bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    keep = ONNXNMSRotatedOp.apply(bboxes_for_nms, scores, iou_threshold)

    if keep_top_k > 0:
        keep = keep[:keep_top_k]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    dets = torch.cat([bboxes, scores[:, None]], 1)

    return dets, labels
