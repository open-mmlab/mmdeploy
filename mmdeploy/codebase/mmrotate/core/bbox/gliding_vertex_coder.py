# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.core.bbox.coder.gliding_vertex_coder'
    '.GVFixCoder.decode')
def gvfixcoder__decode(ctx, self, hbboxes, fix_deltas):
    """Rewriter for GVFixCoder decode, support more dimension input."""

    from mmrotate.core.bbox.transforms import poly2obb
    x1 = hbboxes[..., 0::4]
    y1 = hbboxes[..., 1::4]
    x2 = hbboxes[..., 2::4]
    y2 = hbboxes[..., 3::4]
    w = hbboxes[..., 2::4] - hbboxes[..., 0::4]
    h = hbboxes[..., 3::4] - hbboxes[..., 1::4]

    pred_t_x = x1 + w * fix_deltas[..., 0::4]
    pred_r_y = y1 + h * fix_deltas[..., 1::4]
    pred_d_x = x2 - w * fix_deltas[..., 2::4]
    pred_l_y = y2 - h * fix_deltas[..., 3::4]

    polys = torch.stack(
        [pred_t_x, y1, x2, pred_r_y, pred_d_x, y2, x1, pred_l_y], dim=-1)
    polys = polys.flatten(2)
    rbboxes = poly2obb(polys, self.version)

    return rbboxes
