# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.task_modules.coders.delta_xywh_bbox_coder.'
    'DeltaXYWHBBoxCoder.decode',
    backend='default')
def deltaxywhbboxcoder__decode(self,
                               bboxes,
                               pred_bboxes,
                               max_shape=None,
                               wh_ratio_clip=16 / 1000):
    """Rewrite `decode` of `DeltaXYWHBBoxCoder` for default backend.

    Rewrite this func to call `delta2bbox` directly.

    Args:
        bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
        pred_bboxes (Tensor): Encoded offsets with respect to each roi.
           Has shape (B, N, num_classes * 4) or (B, N, 4) or
           (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
           when rois is a grid of anchors.Offset encoding follows [1]_.
        max_shape (Sequence[int] or torch.Tensor or Sequence[
           Sequence[int]],optional): Maximum bounds for boxes, specifies
           (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
           the max_shape should be a Sequence[Sequence[int]]
           and the length of max_shape should also be B.
        wh_ratio_clip (float, optional): The allowed ratio between
            width and height.

    Returns:
        torch.Tensor: Decoded boxes.
    """
    assert pred_bboxes.size(0) == bboxes.size(0)
    if pred_bboxes.ndim == 3:
        assert pred_bboxes.size(1) == bboxes.size(1)
    from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import \
        delta2bbox
    decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                max_shape, wh_ratio_clip, self.clip_border,
                                self.add_ctr_clamp, self.ctr_clamp)
    return decoded_bboxes


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.task_modules.coders'
    '.delta_xywh_bbox_coder.delta2bbox',
    backend='default')
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):
    """Rewrite `delta2bbox` for default backend.

    Since the need of clip op with dynamic min and max, this function uses
    clip_bboxes function to support dynamic shape.

    Args:
        ctx (ContextCaller): The context with additional information.
        rois (Tensor): Boxes to be transformed. Has shape (N, 4).
        deltas (Tensor): Encoded offsets relative to each roi.
            Has shape (N, num_classes * 4) or (N, 4). Note
            N = num_base_anchors * W * H, when rois is a grid of
            anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates.
            Default (0., 0., 0., 0.).
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates. Default (1., 1., 1., 1.).
        max_shape (tuple[int, int]): Maximum bounds for boxes, specifies
           (H, W). Default None.
        wh_ratio_clip (float): Maximum aspect ratio for boxes. Default
            16 / 1000.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Default True.
        add_ctr_clamp (bool): Whether to add center clamp. When set to True,
            the center of the prediction bounding box will be clamped to
            avoid being too far away from the center of the anchor.
            Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Return:
        bboxes (Tensor): Boxes with shape (N, num_classes * 4) or (N, 4),
            where 4 represent tl_x, tl_y, br_x, br_y.
    """
    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 4))
    denorm_deltas = reshaped_deltas * stds + means

    dxy = denorm_deltas[..., :2]
    dwh = denorm_deltas[..., 2:]

    xy1 = rois[..., None, :2]
    xy2 = rois[..., None, 2:]

    pxy = (xy1 + xy2) * 0.5
    pwh = xy2 - xy1
    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    # Use exp(network energy) to enlarge/shrink each roi
    half_gwh = pwh * dwh.exp() * 0.5
    # Use network energy to shift the center of each roi
    gxy = pxy + dxy_wh

    # Convert center-xy/width/height to top-left, bottom-right
    xy1 = gxy - half_gwh
    xy2 = gxy + half_gwh

    x1 = xy1[..., 0]
    y1 = xy1[..., 1]
    x2 = xy2[..., 0]
    y2 = xy2[..., 1]

    if clip_border and max_shape is not None:
        from mmdeploy.codebase.mmdet.deploy import clip_bboxes
        x1, y1, x2, y2 = clip_bboxes(x1, y1, x2, y2, max_shape)

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
    return bboxes


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.task_modules.coders.'
    'delta_xywh_bbox_coder.delta2bbox',
    backend='ncnn')
def delta2bbox__ncnn(rois,
                     deltas,
                     means=(0., 0., 0., 0.),
                     stds=(1., 1., 1., 1.),
                     max_shape=None,
                     wh_ratio_clip=16 / 1000,
                     clip_border=True,
                     add_ctr_clamp=False,
                     ctr_clamp=32):
    """Rewrite `delta2bbox` for ncnn backend.
    Batch dimension is not supported by ncnn, but supported by pytorch.
    ncnn regards the lowest two dimensions as continuous address with byte
    alignment, so the lowest two dimensions are not absolutely independent.
    Reshape operator with -1 arguments should operates ncnn::Mat with
    dimension >= 3.
    Args:
        ctx (ContextCaller): The context with additional information.
        rois (Tensor): Boxes to be transformed. Has shape (N, 4) or (B, N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (B, N, num_classes * 4) or (B, N, 4) or
            (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
            when rois is a grid of anchors.Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If rois shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    Return:
        bboxes (Tensor): Boxes with shape (B, N, num_classes * 4) or (B, N, 4)
            or (N, num_classes * 4) or (N, 4), where 4 represent tl_x, tl_y,
            br_x, br_y.
    """
    means = deltas.new_tensor(means).view(1, 1, 1, -1).data
    stds = deltas.new_tensor(stds).view(1, 1, 1, -1).data
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 4))
    denorm_deltas = reshaped_deltas * stds + means

    dxy = denorm_deltas[..., :2]
    dwh = denorm_deltas[..., 2:]

    xy1 = rois[..., None, :2]
    xy2 = rois[..., None, 2:]

    pxy = (xy1 + xy2) * 0.5
    pwh = xy2 - xy1
    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    # Use exp(network energy) to enlarge/shrink each roi
    half_gwh = pwh * dwh.exp() * 0.5
    # Use network energy to shift the center of each roi
    gxy = pxy + dxy_wh

    # Convert center-xy/width/height to top-left, bottom-right
    xy1 = gxy - half_gwh
    xy2 = gxy + half_gwh

    x1 = xy1[..., 0]
    y1 = xy1[..., 1]
    x2 = xy2[..., 0]
    y2 = xy2[..., 1]

    if clip_border and max_shape is not None:
        from mmdeploy.codebase.mmdet.deploy import clip_bboxes
        x1, y1, x2, y2 = clip_bboxes(x1, y1, x2, y2, max_shape)

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
    return bboxes
