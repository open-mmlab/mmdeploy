# Copyright (c) OpenMMLab. All rights reserved.
import torch

import mmdeploy
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops import ONNXNMSRotatedOp, TRTBatchedBEVNMSop


def select_nms_index(scores,
                     bboxes,
                     nms_index,
                     keep_top_k,
                     dir_scores=None,
                     attr_scores=None):
    """Transform NMSBEV output.
    Args:
        scores (Tensor): The detection scores of shape
            [N, num_classes, num_boxes].
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 6].
        nms_index (Tensor): NMS output of bounding boxes indexing.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        dir_scores (Tensor): The direction scores of shape
            [N, num_boxes]. Defaults to None.
        attr_scores (Tensor): The attribute scores of shape
            [N, num_boxes]. Defaults to None.
    Returns:
        tuple[Tensor, Tensor, Tensor]: (bbox, scores, labels), `dets` of
            shape [N, num_det, 9] and `scores` of shape [N, num_det] and
            `labels` of shape [N, num_det]
    """
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    # index by nms output
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(0)
    bboxes = bboxes[batch_inds, box_inds, ...].unsqueeze(0)
    labels = cls_inds.unsqueeze(0)
    if dir_scores is not None:
        dir_scores = dir_scores[batch_inds, box_inds].unsqueeze(0)
    if attr_scores is not None:
        attr_scores = attr_scores[batch_inds, box_inds].unsqueeze(0)

    # sort
    is_use_topk = keep_top_k > 0 and (torch.onnx.is_in_onnx_export()
                                      or keep_top_k < scores.shape[1])
    if is_use_topk:
        scores, topk_inds = scores.topk(keep_top_k, dim=1)
    else:
        scores, topk_inds = scores.sort(dim=1, descending=True)
    topk_inds = topk_inds.squeeze(0)
    bboxes = bboxes[:, topk_inds, :]
    labels = labels[:, topk_inds]
    results = (bboxes, scores, labels)

    if dir_scores is not None:
        dir_scores = dir_scores[:, topk_inds]
        results = results + (dir_scores, )
    if attr_scores is not None:
        attr_scores = attr_scores[:, topk_inds]
        results = results + (attr_scores, )
    return results


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet3d.core.post_processing.box3d_nms.'
    '_box3d_multiclass_nms',
    backend='tensorrt')
def box3d_multiclass_nms__tensorrt(
    ctx,
    mlvl_bboxes,
    mlvl_bboxes_for_nms,
    mlvl_scores,
    score_thr,
    nms_thr,
    max_num,
    mlvl_dir_scores=None,
    mlvl_attr_scores=None,
):
    """Multi-class NMS for 3D boxes.

    The IoU used for NMS is defined as the 2D
    IoU between BEV boxes.
    Args:
        mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
            The coordinate system of the BEV boxes is counterclockwise.
        mlvl_scores (torch.Tensor): Multi-level boxes with shape
            (N, C + 1). N is the number of boxes. C is the number of classes.
        score_thr (float): Score threshold to filter boxes with low confidence.
        nms_thr (float): Iou threshold for NMS.
        max_num (int): Maximum number of boxes will be kept.
        mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
            of direction classifier. Defaults to None.
        mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
            of attribute classifier. Defaults to None.
    Returns:
        tuple[torch.Tensor]: Return results after nms, including 3D
            bounding boxes, scores, labels, direction scores (optional),
            attribute scores (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = int(mlvl_scores.shape[-1])
    mlvl_bboxes_for_nms = mlvl_bboxes_for_nms.unsqueeze(2)
    dets, labels, selected = TRTBatchedBEVNMSop.apply(mlvl_bboxes_for_nms,
                                                      mlvl_scores, num_classes,
                                                      -1, max_num, nms_thr,
                                                      score_thr)
    selected = selected.squeeze(0)
    bboxes = mlvl_bboxes[:, selected, :]
    scores = dets[:, :, -1]
    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        dir_scores = mlvl_dir_scores[:, selected]
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        attr_scores = mlvl_attr_scores[:, selected]
        results = results + (attr_scores, )
    return results


def _box3d_multiclass_nms(
    bboxes,
    bboxes_for_nms,
    scores,
    score_thr,
    nms_thr,
    max_num,
    dir_scores=None,
    attr_scores=None,
):
    """NMSBEV for multi-class bboxes.This function helps exporting to onnx with
    batch and multiclass NMSBEV op. It only supports class-agnostic detection
    results. That is, the scores is of shape (N, num_bboxes, num_classes) and
    the boxes is of shape (N, num_boxes, 5).
    Args:
        boxes (Tensor): The 3D bounding boxes of shape [N, num_boxes, 9].
        bboxes_for_nms (Tensor): The 2D bounding boxes of shape
            [N, num_boxes, 5].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        score_thr (float): Score threshold to filter boxes with low confidence.
        nms_thr (float): Iou threshold for NMS.
        max_num (int): Maximum number of boxes will be kept.
        dir_scores (torch.Tensor, optional): The direction scores of shape
        [N, num_boxes]. Defaults to None.
        attr_scores (torch.Tensor, optional): The attribute scores of shape
        [N, num_boxes]. Defaults to None.
    Returns:
        tuple[torch.Tensor]: Return results after nms, including 3D
            bounding boxes, scores, labels, direction scores (optional),
            attribute scores (optional).
    """
    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSRotatedOp.apply(bboxes_for_nms, scores, nms_thr,
                                              score_thr)

    return select_nms_index(scores, bboxes, selected_indices, max_num,
                            dir_scores, attr_scores)


# @FUNCTION_REWRITER.register_rewriter(
#     func_name='mmdet3d.core.post_processing.box3d_multiclass_nms')
def box3d_multiclass_nms(*args, **kwargs):
    """Wrapper function for `_box3d_multiclass_nms`."""
    return mmdeploy.codebase.mmdet3d.core.post_processing.box3d_nms.\
        _box3d_multiclass_nms(*args, **kwargs)
