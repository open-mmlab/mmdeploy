# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.structures.bbox import get_box_tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmrotate.models.task_modules.coders.gliding_vertex_coder'
    '.GVFixCoder.decode')
def gvfixcoder__decode(self, hboxes, fix_deltas):
    """Rewriter for GVFixCoder decode, support more dimension input."""

    assert hboxes.size(
        -1) == 4, f'expect hboxes.size(-1)==4 get {hboxes.size(-1)}.'
    hboxes = get_box_tensor(hboxes)
    x1 = hboxes[..., 0::4]
    y1 = hboxes[..., 1::4]
    x2 = hboxes[..., 2::4]
    y2 = hboxes[..., 3::4]
    w = hboxes[..., 2::4] - hboxes[..., 0::4]
    h = hboxes[..., 3::4] - hboxes[..., 1::4]

    pred_t_x = x1 + w * fix_deltas[..., 0::4]
    pred_r_y = y1 + h * fix_deltas[..., 1::4]
    pred_d_x = x2 - w * fix_deltas[..., 2::4]
    pred_l_y = y2 - h * fix_deltas[..., 3::4]

    polys = torch.stack(
        [pred_t_x, y1, x2, pred_r_y, pred_d_x, y2, x1, pred_l_y], dim=-1)
    polys = polys.flatten(2)

    return polys
