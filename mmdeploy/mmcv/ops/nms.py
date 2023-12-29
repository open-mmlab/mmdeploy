# Copyright (c) OpenMMLab. All rights reserved.
import torch
from packaging import version
from torch import Tensor
from torch.onnx import symbolic_helper as sym_help

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import IR, is_dynamic_batch
from mmdeploy.utils.constants import Backend
from .nms_match import multiclass_nms_match
from .nms_rotated import multiclass_nms_rotated


class ONNXNMSop(torch.autograd.Function):
    """Create onnx::NonMaxSuppression op.

    NMS in mmcv only supports one class with no batch info. This class assists
    in exporting NMS of ONNX's definition.
    """

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor,
                max_output_boxes_per_class: int, iou_threshold: float,
                score_threshold: float) -> Tensor:
        """Get NMS output indices.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            Tensor: Selected indices of boxes. 2-D tensor of shape
            (num_selected_indices, 3) with each row of
            [batch_index, class_index, box_index].
        """
        from mmcv.ops import nms
        batch_size, num_class, _ = scores.shape

        score_threshold = float(score_threshold)
        iou_threshold = float(iou_threshold)
        indices = []
        for batch_id in range(batch_size):
            for cls_id in range(num_class):
                _boxes = boxes[batch_id, ...]
                # score_threshold=0 requires scores to be contiguous
                _scores = scores[batch_id, cls_id, ...].contiguous()
                _, box_inds = nms(
                    _boxes,
                    _scores,
                    iou_threshold,
                    offset=0,
                    score_threshold=score_threshold,
                    max_num=max_output_boxes_per_class)
                batch_inds = torch.zeros_like(box_inds) + batch_id
                cls_inds = torch.zeros_like(box_inds) + cls_id
                indices.append(
                    torch.stack([batch_inds, cls_inds, box_inds], dim=-1))
        indices = torch.cat(indices)
        return indices

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor,
                 max_output_boxes_per_class: int, iou_threshold: float,
                 score_threshold: float):
        """Symbolic function for onnx::NonMaxSuppression.

        Args:
            g (Graph): The traced onnx graph.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            max_output_boxes_per_class (int): Maximum number of output
                boxes per class of nms.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            NonMaxSuppression op for onnx.
        """
        if not sym_help._is_value(max_output_boxes_per_class):
            max_output_boxes_per_class = g.op(
                'Constant',
                value_t=torch.tensor(
                    max_output_boxes_per_class, dtype=torch.long))

        if not sym_help._is_value(iou_threshold):
            iou_threshold = g.op(
                'Constant',
                value_t=torch.tensor([iou_threshold], dtype=torch.float))

        if not sym_help._is_value(score_threshold):
            score_threshold = g.op(
                'Constant',
                value_t=torch.tensor([score_threshold], dtype=torch.float))
        return g.op('NonMaxSuppression', boxes, scores,
                    max_output_boxes_per_class, iou_threshold, score_threshold)


class TRTBatchedNMSop(torch.autograd.Function):
    """Create mmdeploy::TRTBatchedNMS op for TensorRT backend.

    NMS in ONNX supports dynamic outputs. This class helps replace
    onnx::NonMaxSuppression with mmdeploy::TRTBatchedNMS.
    """

    @staticmethod
    def forward(ctx,
                boxes: Tensor,
                scores: Tensor,
                num_classes: int,
                pre_topk: int,
                after_topk: int,
                iou_threshold: float,
                score_threshold: float,
                background_label_id: int = -1,
                return_index: bool = False):
        """Forward of batched nms.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            num_classes (int): MThe number of classes in the network.
            pre_topk (int): The number of bounding boxes to be fed into
                the NMS step.
            after_topk (int): The number of total bounding boxes to be kept
                per-image after the NMS step. Should be less than or equal
                to the pre_topk value.
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.
            background_label_id (int): The label ID for the background class.
                If there is no background class, set it to -1.

        Returns:
            Tensor: Selected indices of boxes. 2-D tensor of shape
            (num_selected_indices, 3) with each row of
            [batch_index, class_index, box_index]. Note it is generated
            randomly to make it exportable to onnx.
        """
        batch_size, num_boxes, num_classes = scores.shape

        out_boxes = min(num_boxes, after_topk)
        ret = (torch.rand(batch_size, out_boxes, 5).to(scores.device),
               torch.randint(0, num_classes,
                             (batch_size, out_boxes)).to(scores.device))
        if return_index:
            ret = ret + (torch.randint(
                0, out_boxes, (batch_size, out_boxes)).to(scores.device), )
        return ret

    @staticmethod
    def symbolic(g,
                 boxes: Tensor,
                 scores: Tensor,
                 num_classes: int,
                 pre_topk: int,
                 after_topk: int,
                 iou_threshold: float,
                 score_threshold: float,
                 background_label_id: int = -1,
                 return_index: bool = False):
        """Symbolic function for mmdeploy::TRTBatchedNMS."""
        return g.op(
            'mmdeploy::TRTBatchedNMS',
            boxes,
            scores,
            num_classes_i=num_classes,
            background_label_id_i=background_label_id,
            iou_threshold_f=iou_threshold,
            score_threshold_f=score_threshold,
            topk_i=pre_topk,
            keep_topk_i=after_topk,
            is_normalized_i=False,
            clip_boxes_i=False,
            return_index_i=return_index,
            outputs=3 if return_index else 2)


def _select_nms_index(scores: torch.Tensor,
                      boxes: torch.Tensor,
                      nms_index: torch.Tensor,
                      batch_size: int,
                      keep_top_k: int = -1,
                      pre_inds: torch.Tensor = None,
                      output_index: bool = False):
    """Transform NMS output.

    Args:
        scores (Tensor): The detection scores of shape
            [N, num_classes, num_boxes].
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        nms_index (Tensor): NMS output of bounding boxes indexing.
        batch_size (int): Batch size of the input image.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        pre_inds (Tensor): The pre-topk indices of boxes before nms.
            Defaults to None.
        return_index (bool): Whether to return indices of original bboxes.
            Defaults to False.

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
    if output_index and pre_inds is not None:
        # batch all
        pre_inds = pre_inds[batch_inds, box_inds]
        pre_inds = pre_inds.unsqueeze(0).repeat(batch_size, 1)
        pre_inds = pre_inds.where((batch_inds == batch_template.unsqueeze(1)),
                                  pre_inds.new_zeros(1))
        pre_inds = torch.cat((pre_inds, -pre_inds.new_ones((N, 1))), 1)
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
    if output_index:
        if pre_inds is not None:
            topk_inds = pre_inds[topk_batch_inds, topk_inds, ...]
        return batched_dets, batched_labels, topk_inds
    # slice and recover the tensor
    return batched_dets, batched_labels


def _multiclass_nms(boxes: Tensor,
                    scores: Tensor,
                    max_output_boxes_per_class: int = 1000,
                    iou_threshold: float = 0.5,
                    score_threshold: float = 0.05,
                    pre_top_k: int = -1,
                    keep_top_k: int = -1,
                    output_index: bool = False):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op. It
    only supports class-agnostic detection results. That is, the scores is of
    shape (N, num_bboxes, num_classes) and the boxes is of shape (N, num_boxes,
    4).
    """
    if version.parse(torch.__version__) < version.parse('1.13.0'):
        max_output_boxes_per_class = torch.LongTensor(
            [max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]
    topk_inds = None
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        batch_inds = torch.arange(
            batch_size, device=scores.device).view(-1, 1).long()
        boxes = boxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSop.apply(boxes, scores,
                                       max_output_boxes_per_class,
                                       iou_threshold, score_threshold)

    return _select_nms_index(
        scores,
        boxes,
        selected_indices,
        batch_size,
        keep_top_k=keep_top_k,
        pre_inds=topk_inds,
        output_index=output_index)


def _multiclass_nms_single(boxes: Tensor,
                           scores: Tensor,
                           max_output_boxes_per_class: int = 1000,
                           iou_threshold: float = 0.5,
                           score_threshold: float = 0.05,
                           pre_top_k: int = -1,
                           keep_top_k: int = -1,
                           output_index: bool = False):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    Single batch nms could be optimized.
    """
    if version.parse(torch.__version__) < version.parse('1.13.0'):
        max_output_boxes_per_class = torch.LongTensor(
            [max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)

    # pre topk
    pre_topk_inds = None
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.squeeze(0).topk(pre_top_k)
        boxes = boxes[:, topk_inds, :]
        scores = scores[:, topk_inds, :]
        pre_topk_inds = topk_inds

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

    if output_index:
        bbox_index = box_inds.unsqueeze(0)
        if pre_top_k > 0:
            bbox_index = pre_topk_inds[None, box_inds]
        # pad index to keep same dim as dets and labels
        bbox_index = torch.cat([bbox_index, -bbox_index.new_ones((1, 1))], 1)
        if keep_top_k > 0:
            bbox_index = bbox_index[:, topk_inds]
        return dets, labels, bbox_index
    else:
        return dets, labels


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.mmcv.ops.nms._multiclass_nms')
def multiclass_nms__default(boxes: Tensor,
                            scores: Tensor,
                            max_output_boxes_per_class: int = 1000,
                            iou_threshold: float = 0.5,
                            score_threshold: float = 0.05,
                            pre_top_k: int = -1,
                            keep_top_k: int = -1,
                            output_index: bool = False):
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
    ctx = FUNCTION_REWRITER.get_context()
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
            keep_top_k=keep_top_k,
            output_index=output_index)
    else:
        return ctx.origin_func(
            boxes,
            scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k,
            output_index=output_index)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.mmcv.ops.nms._multiclass_nms', backend='tensorrt')
def multiclass_nms_static(boxes: Tensor,
                          scores: Tensor,
                          max_output_boxes_per_class: int = 1000,
                          iou_threshold: float = 0.5,
                          score_threshold: float = 0.05,
                          pre_top_k: int = -1,
                          keep_top_k: int = -1,
                          output_index: bool = False):
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
    nms_output = TRTBatchedNMSop.apply(
        boxes,
        scores,
        int(scores.shape[-1]),
        pre_top_k,
        keep_top_k,
        iou_threshold,
        score_threshold,
        -1,
        output_index,
    )

    dets = nms_output[0]
    labels = nms_output[1]
    box_index = None if len(nms_output) <= 2 else nms_output[2]

    # retain shape info
    batch_size = boxes.size(0)

    dets_shape = dets.shape
    label_shape = labels.shape
    dets = dets.reshape([batch_size, *dets_shape[1:]])
    labels = labels.reshape([batch_size, *label_shape[1:]])
    if output_index:
        return dets, labels, box_index
    return dets, labels


@mark(
    'multiclass_nms',
    inputs=['boxes', 'scores'],
    outputs=['dets', 'labels', 'index'])
def multiclass_nms(boxes: Tensor,
                   scores: Tensor,
                   max_output_boxes_per_class: int = 1000,
                   iou_threshold: float = 0.5,
                   score_threshold: float = 0.05,
                   pre_top_k: int = -1,
                   keep_top_k: int = -1,
                   output_index: bool = False,
                   nms_type='nms'):
    """Apis for multiclass nms."""
    if nms_type == 'nms':
        return _multiclass_nms(
            boxes,
            scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k,
            output_index=output_index)
    elif nms_type == 'nms_rotated':
        return multiclass_nms_rotated(
            boxes,
            scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)
    elif nms_type == 'nms_match':
        return multiclass_nms_match(
            boxes,
            scores,
            max_output_boxes_per_class=max_output_boxes_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pre_top_k=pre_top_k,
            keep_top_k=keep_top_k)
    else:
        raise NotImplementedError(f'Unsupported nms type: {nms_type}.')


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.mmcv.ops.nms._multiclass_nms',
    backend=Backend.COREML.value)
def multiclass_nms__coreml(boxes: Tensor,
                           scores: Tensor,
                           max_output_boxes_per_class: int = 1000,
                           iou_threshold: float = 0.5,
                           score_threshold: float = 0.05,
                           pre_top_k: int = -1,
                           keep_top_k: int = -1,
                           output_index: bool = False):
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
    boxes, scores, box_index, _ = coreml_nms(
        boxes, scores, iou_threshold, score_threshold,
        min(keep_top_k, max_output_boxes_per_class))

    scores, labels = scores.max(-1)
    boxes = _xywh2xyxy(boxes)
    dets = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)

    if output_index:
        return dets, labels, box_index
    return dets, labels


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.mmcv.ops.nms._multiclass_nms', ir=IR.TORCHSCRIPT)
def multiclass_nms__torchscript(boxes: Tensor,
                                scores: Tensor,
                                max_output_boxes_per_class: int = 1000,
                                iou_threshold: float = 0.5,
                                score_threshold: float = 0.05,
                                pre_top_k: int = -1,
                                keep_top_k: int = -1,
                                output_index=False):
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
    pre_topk_inds = None
    # pre-topk
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        pre_topk_inds = topk_inds
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
    return _select_nms_index(
        scores,
        boxes,
        keeps,
        batch_size,
        keep_top_k=keep_top_k,
        pre_inds=pre_topk_inds,
        output_index=output_index)


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
    func_name='mmdeploy.mmcv.ops.nms._multiclass_nms', backend='ascend')
def multiclass_nms__ascend(boxes: Tensor,
                           scores: Tensor,
                           max_output_boxes_per_class: int = 1000,
                           iou_threshold: float = 0.5,
                           score_threshold: float = 0.05,
                           pre_top_k: int = -1,
                           keep_top_k: int = -1,
                           output_index: bool = False):
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
    assert not output_index, 'output_index is not supported on this backend.'
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    keep_top_k = max_output_boxes_per_class if keep_top_k < 0 else min(
        max_output_boxes_per_class, keep_top_k)
    nmsed_boxes, nmsed_scores, nmsed_classes, _ = AscendBatchNMSOp.apply(
        boxes, scores, score_threshold, iou_threshold, keep_top_k, keep_top_k)

    dets = torch.cat([nmsed_boxes, nmsed_scores.unsqueeze(2)], dim=-1)
    return dets, nmsed_classes.int()
