# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.layers.matrix_nms.mask_matrix_nms')
def mask_matrix_nms__default(masks,
                             labels,
                             scores,
                             filter_thr=-1,
                             nms_pre=-1,
                             max_num=-1,
                             kernel='gaussian',
                             sigma=2.0,
                             mask_area=None):
    """Matrix NMS for multi-class masks.

    Args:
        masks (Tensor): Has shape (num_instances, h, w)
        labels (Tensor): Labels of corresponding masks,
            has shape (num_instances,).
        scores (Tensor): Mask scores of corresponding masks,
            has shape (num_instances).
        filter_thr (float): Score threshold to filter the masks
            after matrix nms. Default: -1, which means do not
            use filter_thr.
        nms_pre (int): The max number of instances to do the matrix nms.
            Default: -1, which means do not use nms_pre.
        max_num (int, optional): If there are more than max_num masks after
            matrix, only top max_num will be kept. Default: -1, which means
            do not use max_num.
        kernel (str): 'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        mask_area (Tensor): The sum of seg_masks.

    Returns:
        tuple(Tensor): Processed mask results.

            - scores (Tensor): Updated scores, has shape (n,).
            - labels (Tensor): Remained labels, has shape (n,).
            - masks (Tensor): Remained masks, has shape (n, w, h).
            - keep_inds (Tensor): The indices number of
                the remaining mask in the input mask, has shape (n,).
    """
    assert len(labels) == len(masks) == len(scores)
    assert len(masks) == len(mask_area)
    # sort and keep top nms_pre
    nms_pre = max(0, nms_pre)
    if nms_pre <= 0:
        nms_pre = scores.shape[0]
    scores, sort_inds = torch.topk(scores, nms_pre)

    keep_inds = sort_inds
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]
    num_masks = labels.size(0)
    flatten_masks = masks.reshape(num_masks, -1).float()
    # inter.
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.unsqueeze(1)
    total_area = expanded_mask_area + expanded_mask_area.transpose(
        1, 0) - inter_matrix
    total_mask = total_area > 0
    total_area = total_area.where(total_mask, total_area.new_ones(1))
    iou_matrix = (inter_matrix / total_area).triu(diagonal=1)
    expanded_labels = labels.unsqueeze(1)
    label_matrix = expanded_labels == expanded_labels.transpose(1, 0)

    # iou decay
    decay_iou = iou_matrix.where(label_matrix, iou_matrix.new_zeros(1))

    # iou compensation
    compensate_iou, _ = decay_iou.max(0)
    compensate_iou = compensate_iou.expand(num_masks,
                                           num_masks).transpose(1, 0)

    # calculate the decay_coefficient
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError(
            f'{kernel} kernel is not supported in matrix nms!')
    # update the score.
    scores = scores * decay_coefficient

    keep = scores >= filter_thr
    scores = scores.where(keep, scores.new_zeros(1))

    # sort and keep top max_num
    scores, sort_inds = torch.topk(scores, max(max_num, 0))
    keep_inds = keep_inds[sort_inds]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds
