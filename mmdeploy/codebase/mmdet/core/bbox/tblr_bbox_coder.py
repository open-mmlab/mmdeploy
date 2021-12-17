# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.core.bbox.coder.tblr_bbox_coder.tblr2bboxes',
    backend='default')
def tblr2bboxes(ctx,
                priors,
                tblr,
                normalizer=4.0,
                normalize_by_wh=True,
                max_shape=None,
                clip_border=True):
    """Rewrite `tblr2bboxes` for default backend.

    Since the need of clip op with dynamic min and max, this function uses
    clip_bboxes function to support dynamic shape.

    Args:
        ctx (ContextCaller): The context with additional information.
        priors (Tensor): Prior boxes in point form (x0, y0, x1, y1)
          Shape: (N,4) or (B, N, 4).
        tblr (Tensor): Coords of network output in tblr form
          Shape: (N, 4) or (B, N, 4).
        normalizer (Sequence[float] | float): Normalization parameter of
          encoded boxes. By list, it represents the normalization factors at
          tblr dims. By float, it is the unified normalization factor at all
          dims. Default: 4.0
        normalize_by_wh (bool): Whether the tblr coordinates have been
          normalized by the side length (wh) of prior bboxes.
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Return:
        bboxes (Tensor): Boxes with shape (N, 4) or (B, N, 4)
    """
    if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    assert priors.size(0) == tblr.size(0)
    if priors.ndim == 3:
        assert priors.size(1) == tblr.size(1)

    loc_decode = tblr * normalizer
    prior_centers = (priors[..., 0:2] + priors[..., 2:4]) / 2
    if normalize_by_wh:
        wh = priors[..., 2:4] - priors[..., 0:2]

        w, h = torch.split(wh, 1, dim=-1)
        # Inplace operation with slice would fail for exporting to ONNX
        th = h * loc_decode[..., :2]  # tb
        tw = w * loc_decode[..., 2:]  # lr
        loc_decode = torch.cat([th, tw], dim=-1)
    top, bottom, left, right = loc_decode.split((1, 1, 1, 1), dim=-1)
    xmin = prior_centers[..., 0].unsqueeze(-1) - left
    xmax = prior_centers[..., 0].unsqueeze(-1) + right
    ymin = prior_centers[..., 1].unsqueeze(-1) - top
    ymax = prior_centers[..., 1].unsqueeze(-1) + bottom

    if clip_border and max_shape is not None:
        from mmdeploy.codebase.mmdet.deploy import clip_bboxes
        xmin, ymin, xmax, ymax = clip_bboxes(xmin, ymin, xmax, ymax, max_shape)
    bboxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1).view(priors.size())

    return bboxes
