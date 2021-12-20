# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.core.bbox.coder.delta_xywh_bbox_coder.'
    'DeltaXYWHBBoxCoder.decode',
    backend='default')
def deltaxywhbboxcoder__decode(ctx,
                               self,
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
    from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
    decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                max_shape, wh_ratio_clip, self.clip_border,
                                self.add_ctr_clamp, self.ctr_clamp)
    return decoded_bboxes


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.core.bbox.coder.delta_xywh_bbox_coder.delta2bbox',  # noqa
    backend='default')
def delta2bbox(ctx,
               rois,
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
    means = deltas.new_tensor(means).view(1,
                                          -1).repeat(1,
                                                     deltas.size(-1) // 4)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(-1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[..., 0::4]
    dy = denorm_deltas[..., 1::4]
    dw = denorm_deltas[..., 2::4]
    dh = denorm_deltas[..., 3::4]

    x1, y1 = rois[..., 0], rois[..., 1]
    x2, y2 = rois[..., 2], rois[..., 3]
    # Compute center of each roi
    px = ((x1 + x2) * 0.5).unsqueeze(-1).expand_as(dx)
    py = ((y1 + y2) * 0.5).unsqueeze(-1).expand_as(dy)
    # Compute width/height of each roi
    pw = (x2 - x1).unsqueeze(-1).expand_as(dw)
    ph = (y2 - y1).unsqueeze(-1).expand_as(dh)

    dx_width = pw * dx
    dy_height = ph * dy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    if clip_border and max_shape is not None:
        from mmdeploy.codebase.mmdet.deploy import clip_bboxes
        x1, y1, x2, y2 = clip_bboxes(x1, y1, x2, y2, max_shape)

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
    return bboxes


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.core.bbox.coder.delta_xywh_bbox_coder.delta2bbox',  # noqa
    backend='ncnn')
def delta2bbox__ncnn(ctx,
                     rois,
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
    NCNN regards the lowest two dimensions as continuous address with byte
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
    means = deltas.new_tensor(means).view(1, 1,
                                          -1).repeat(1, deltas.size(-2),
                                                     deltas.size(-1) // 4).data
    stds = deltas.new_tensor(stds).view(1, 1,
                                        -1).repeat(1, deltas.size(-2),
                                                   deltas.size(-1) // 4).data
    denorm_deltas = deltas * stds + means
    if denorm_deltas.shape[-1] == 4:
        dx = denorm_deltas[..., 0:1]
        dy = denorm_deltas[..., 1:2]
        dw = denorm_deltas[..., 2:3]
        dh = denorm_deltas[..., 3:4]
    else:
        dx = denorm_deltas[..., 0::4]
        dy = denorm_deltas[..., 1::4]
        dw = denorm_deltas[..., 2::4]
        dh = denorm_deltas[..., 3::4]

    x1, y1 = rois[..., 0:1], rois[..., 1:2]
    x2, y2 = rois[..., 2:3], rois[..., 3:4]

    # Compute center of each roi
    px = (x1 + x2) * 0.5
    py = (y1 + y2) * 0.5
    # Compute width/height of each roi
    pw = x2 - x1
    ph = y2 - y1

    # do not use expand unless necessary
    # since expand is a custom ops
    if px.shape[-1] != 4:
        px = px.expand_as(dx)
    if py.shape[-1] != 4:
        py = py.expand_as(dy)
    if pw.shape[-1] != 4:
        pw = pw.expand_as(dw)
    if px.shape[-1] != 4:
        ph = ph.expand_as(dh)

    dx_width = pw * dx
    dy_height = ph * dy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    if clip_border and max_shape is not None:
        from mmdeploy.codebase.mmdet.deploy import clip_bboxes
        x1, y1, x2, y2 = clip_bboxes(x1, y1, x2, y2, max_shape)

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
    return bboxes
