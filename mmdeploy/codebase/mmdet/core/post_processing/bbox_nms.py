# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops.nms import ONNXNMSop, TRTBatchedNMSop
from mmdeploy.utils import IR, is_dynamic_batch
from mmdeploy.utils.constants import Backend


def select_nms_index(scores: torch.Tensor,
                     boxes: torch.Tensor,
                     nms_index: torch.Tensor,
                     batch_size: int,
                     keep_top_k: int = -1):
    """Transform NMS output.

    Args:
        scores (Tensor): The detection scores of shape
            [N, num_classes, num_boxes].
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        nms_index (Tensor): NMS output of bounding boxes indexing.
        batch_size (int): Batch size of the input image.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    # index by nms output
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)

    # batch all
    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(
        0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    batched_dets = batched_dets.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_dets.new_zeros(1))

    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where(
        (batch_inds == batch_template.unsqueeze(1)),
        batched_labels.new_ones(1) * -1)

    N = batched_dets.shape[0]

    # expand tensor to eliminate [0, ...] tensor
    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))),
                             1)
    batched_labels = torch.cat((batched_labels, batched_labels.new_zeros(
        (N, 1))), 1)

    # sort
    is_use_topk = keep_top_k > 0 and \
        (torch.onnx.is_in_onnx_export() or keep_top_k < batched_dets.shape[1])
    if is_use_topk:
        _, topk_inds = batched_dets[:, :, -1].topk(keep_top_k, dim=1)
    else:
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(
        batch_size, dtype=topk_inds.dtype,
        device=topk_inds.device).view(-1, 1)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]

    # slice and recover the tensor
    return batched_dets, batched_labels


def _multiclass_nms(boxes: Tensor,
                    scores: Tensor,
                    max_output_boxes_per_class: int = 1000,
                    iou_threshold: float = 0.5,
                    score_threshold: float = 0.05,
                    pre_top_k: int = -1,
                    keep_top_k: int = -1):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op. It
    only supports class-agnostic detection results. That is, the scores is of
    shape (N, num_bboxes, num_classes) and the boxes is of shape (N, num_boxes,
    4).
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]

    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        batch_inds = torch.arange(batch_size).view(-1, 1).long()
        boxes = boxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSop.apply(boxes, scores,
                                       max_output_boxes_per_class,
                                       iou_threshold, score_threshold)

    dets, labels = select_nms_index(
        scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)

    return dets, labels


def _multiclass_nms_single(boxes: Tensor,
                           scores: Tensor,
                           max_output_boxes_per_class: int = 1000,
                           iou_threshold: float = 0.5,
                           score_threshold: float = 0.05,
                           pre_top_k: int = -1,
                           keep_top_k: int = -1):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    Single batch nms could be optimized.
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)

    # pre topk
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.squeeze(0).topk(pre_top_k)
        boxes = boxes[:, topk_inds, :]
        scores = scores[:, topk_inds, :]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSop.apply(boxes, scores,
                                       max_output_boxes_per_class,
                                       iou_threshold, score_threshold)

    cls_inds = selected_indices[:, 1]
    box_inds = selected_indices[:, 2]

    scores = scores[:, cls_inds, box_inds].unsqueeze(2)
    boxes = boxes[:, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=2)
    labels = cls_inds.unsqueeze(0)

    # pad
    dets = torch.cat((dets, dets.new_zeros((1, 1, 5))), 1)
    labels = torch.cat((labels, labels.new_zeros((1, 1))), 1)

    # topk or sort
    is_use_topk = keep_top_k > 0 and \
        (torch.onnx.is_in_onnx_export() or keep_top_k < dets.shape[1])
    if is_use_topk:
        _, topk_inds = dets[:, :, -1].topk(keep_top_k, dim=1)
    else:
        _, topk_inds = dets[:, :, -1].sort(dim=1, descending=True)
    topk_inds = topk_inds.squeeze(0)
    dets = dets[:, topk_inds, ...]
    labels = labels[:, topk_inds, ...]

    return dets, labels


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.core.post_processing'
    '.bbox_nms._multiclass_nms')
def multiclass_nms__default(ctx,
                            boxes: Tensor,
                            scores: Tensor,
                            max_output_boxes_per_class: int = 1000,
                            iou_threshold: float = 0.5,
                            score_threshold: float = 0.05,
                            pre_top_k: int = -1,
                            keep_top_k: int = -1):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op.
    It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 4).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    deploy_cfg = ctx.cfg
    batch_size = boxes.size(0)
    if not is_dynamic_batch(deploy_cfg) and batch_size == 1:
        return _multiclass_nms_single(
            boxes,
            scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)
    else:
        return ctx.origin_func(
            boxes,
            scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.core.post_processing'
    '.bbox_nms._multiclass_nms',
    backend='tensorrt')
def multiclass_nms_static(ctx,
                          boxes: Tensor,
                          scores: Tensor,
                          max_output_boxes_per_class: int = 1000,
                          iou_threshold: float = 0.5,
                          score_threshold: float = 0.05,
                          pre_top_k: int = -1,
                          keep_top_k: int = -1):
    """Wrapper for `multiclass_nms` with TensorRT.

    Args:
        ctx (ContextCaller): The context with additional information.
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    keep_top_k = max_output_boxes_per_class if keep_top_k < 0 else min(
        max_output_boxes_per_class, keep_top_k)
    dets, labels = TRTBatchedNMSop.apply(boxes, scores, int(scores.shape[-1]),
                                         pre_top_k, keep_top_k, iou_threshold,
                                         score_threshold, -1)

    # retain shape info
    batch_size = boxes.size(0)

    dets_shape = dets.shape
    label_shape = labels.shape
    dets = dets.reshape([batch_size, *dets_shape[1:]])
    labels = labels.reshape([batch_size, *label_shape[1:]])
    return dets, labels


@mark('multiclass_nms', inputs=['boxes', 'scores'], outputs=['dets', 'labels'])
def multiclass_nms(*args, **kwargs):
    """Wrapper function for `_multiclass_nms`."""
    return _multiclass_nms(*args, **kwargs)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.core.post_processing'
    '.bbox_nms._multiclass_nms',
    backend=Backend.COREML.value)
def multiclass_nms__coreml(ctx,
                           boxes: Tensor,
                           scores: Tensor,
                           max_output_boxes_per_class: int = 1000,
                           iou_threshold: float = 0.5,
                           score_threshold: float = 0.05,
                           pre_top_k: int = -1,
                           keep_top_k: int = -1):
    """rewrite for coreml batched nms.

    Use coreml_nms from custom ops.
    """

    # load custom nms
    from mmdeploy.backend.torchscript import get_ops_path, ops_available
    assert ops_available(), 'coreml require custom torchscript ops support.'
    torch.ops.load_library(get_ops_path())
    try:
        coreml_nms = torch.ops.mmdeploy.coreml_nms
    except Exception:
        raise Exception(
            'Can not use coreml_nms. Please build torchscript custom ops.')

    batch_size = scores.shape[0]
    assert batch_size == 1, 'batched nms is not supported for now.'

    # pre-topk
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        boxes = boxes[:, topk_inds.squeeze(), ...]
        scores = scores[:, topk_inds.squeeze(), ...]

    def _xyxy2xywh(boxes):
        xy0 = boxes[..., :2]
        xy1 = boxes[..., 2:]
        xy = (xy0 + xy1) / 2
        wh = xy1 - xy0
        return torch.cat([xy, wh], dim=-1)

    def _xywh2xyxy(boxes):
        xy = boxes[..., :2]
        half_wh = boxes[..., 2:] / 2
        return torch.cat([xy - half_wh, xy + half_wh], dim=-1)

    boxes = _xyxy2xywh(boxes)
    keep_top_k = keep_top_k if keep_top_k > 0 else max_output_boxes_per_class
    boxes, scores, _, _ = coreml_nms(
        boxes, scores, iou_threshold, score_threshold,
        min(keep_top_k, max_output_boxes_per_class))

    scores, labels = scores.max(-1)
    boxes = _xywh2xyxy(boxes)
    dets = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)

    return dets, labels


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.core.post_processing'
    '.bbox_nms._multiclass_nms',
    ir=IR.TORCHSCRIPT)
def multiclass_nms__torchscript(ctx,
                                boxes: Tensor,
                                scores: Tensor,
                                max_output_boxes_per_class: int = 1000,
                                iou_threshold: float = 0.5,
                                score_threshold: float = 0.05,
                                pre_top_k: int = -1,
                                keep_top_k: int = -1):
    """rewrite for torchscript batched nms.

    Use batched_nms from torchvision instead of custom nms.
    """
    # TODO: simplify inference for non-batch model
    from torchvision.ops import batched_nms
    batch_size = scores.shape[0]
    num_boxes = scores.shape[1]
    num_classes = scores.shape[2]
    box_per_cls = len(boxes.shape) == 4
    scores = torch.where(scores > score_threshold, scores, scores.new_zeros(1))

    # pre-topk
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        batch_inds = torch.arange(batch_size).view(-1, 1).long()
        boxes = boxes[batch_inds, topk_inds, ...]
        scores = scores[batch_inds, topk_inds, :]
        num_boxes = scores.shape[1]

    idxs = torch.arange(0, batch_size, device=scores.device).unsqueeze(1)
    idxs = idxs.repeat(1, num_boxes).view(-1)

    keeps = [None] * num_classes
    for cls_id in range(num_classes):
        box = boxes if not box_per_cls else boxes[:, :, cls_id, :]
        score = scores[:, :, cls_id]
        box = box.view(-1, 4)
        score = score.view(-1)
        box_keep = batched_nms(box, score, idxs, iou_threshold=iou_threshold)
        box_keep = box_keep[:max_output_boxes_per_class * batch_size]
        batch_keep = idxs[box_keep]
        cls_keep = torch.ones_like(box_keep) * cls_id
        box_keep = box_keep - batch_keep * num_boxes
        keeps[cls_id] = torch.stack([batch_keep, cls_keep, box_keep], dim=1)

    keeps = torch.cat(keeps)
    scores = scores.permute(0, 2, 1)
    dets, labels = select_nms_index(
        scores, boxes, keeps, batch_size, keep_top_k=keep_top_k)

    return dets, labels


class AscendBatchNMSOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bboxes: torch.Tensor, scores: torch.Tensor,
                score_threshold: float, iou_threshold: float,
                max_size_per_class: int, max_total_size: int):
        """Dummy nms forward
        Args:
        boxes (torch.Tensor): boxes in shape (batch, N, C, 4).
        scores (torch.Tensor): scores in shape (batch, N, C).
        score_threshold (float): the score threshold.
        iou_threshold (float): the iou threshold.
        max_size_per_class (int): max size per class.
        max_total_size (int): max total size.

        Returns:
            (torch.Tensor): boxes,(1, N, 4)
            (torch.Tensor): scores,(1, N)
            (torch.Tensor): classes,(1, N)
            (torch.Tensor): num_dets,(1,)
        """

        # Python implementation for onnx export
        nmsed_boxes = bboxes[:, :max_total_size, 0, :]
        nmsed_scores = scores[:, :max_total_size, 0]
        nmsed_classes = torch.arange(max_total_size, dtype=torch.long)
        nmsed_num = torch.Tensor([max_total_size])

        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num

    @staticmethod
    def symbolic(g, bboxes, scores, score_thr, iou_thr, max_size_p_class,
                 max_t_size):
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = g.op(
            'mmdeploy::BatchMultiClassNMS',
            bboxes,
            scores,
            score_threshold_f=score_thr,
            iou_threshold_f=iou_thr,
            max_size_per_class_i=max_size_p_class,
            max_total_size_i=max_t_size,
            outputs=4)
        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.core.post_processing'
    '.bbox_nms._multiclass_nms',
    backend='ascend')
def multiclass_nms__ascend(ctx,
                           boxes: Tensor,
                           scores: Tensor,
                           max_output_boxes_per_class: int = 1000,
                           iou_threshold: float = 0.5,
                           score_threshold: float = 0.05,
                           pre_top_k: int = -1,
                           keep_top_k: int = -1):
    """Wrapper for `multiclass_nms` with Ascend.

    Args:
        ctx (ContextCaller): The context with additional information.
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    keep_top_k = max_output_boxes_per_class if keep_top_k < 0 else min(
        max_output_boxes_per_class, keep_top_k)
    nmsed_boxes, nmsed_scores, nmsed_classes, _ = AscendBatchNMSOp.apply(
        boxes, scores, score_threshold, iou_threshold, keep_top_k, keep_top_k)

    dets = torch.cat([nmsed_boxes, nmsed_scores.unsqueeze(2)], dim=-1)
    return dets, nmsed_classes
