# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.backbones.csp_darknet.Focus.forward')
def focus__forward__default(ctx, self, x):
    """Rewrite forward function of Focus class.

    Replace slice with transpose.
    """
    # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1, 2, W)
    x = x.reshape(B, C, x.shape[2], 2, -1, 2)
    half_H = x.shape[2]
    half_W = x.shape[4]
    x = x.permute(0, 5, 3, 1, 2, 4)
    x = x.reshape(B, C * 4, half_H, half_W)

    return self.conv(x)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.backbones.csp_darknet.Focus.forward',
    backend='ncnn')
def focus__forward__ncnn(ctx, self, x):
    """Rewrite forward function of Focus class for ncnn.

    Focus width and height information into channel space. ncnn does not
    support slice operator which step greater than 1, so we use another
    way to implement.

    Args:
        x (Tensor): The input tensor with shape (N, C, H, W).

    Returns:
        x (Tensor): The calculated tensor with shape (N, 4*C, H//2, W//2).
    """
    batch_size, c, h, w = x.shape
    assert h % 2 == 0 and w % 2 == 0, f'focus for yolox needs even feature\
        height and width, got {(h, w)}.'

    x = x.reshape(batch_size, c * h, 1, w)
    _b, _c, _h, _w = x.shape
    g = _c // 2
    # fuse to ncnn's shufflechannel
    x = x.view(_b, g, 2, _h, _w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(_b, -1, _h, _w)

    x = x.reshape(_b, c * h * w, 1, 1)

    _b, _c, _h, _w = x.shape
    g = _c // 2
    # fuse to ncnn's shufflechannel
    x = x.view(_b, g, 2, _h, _w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(_b, -1, _h, _w)

    x = x.reshape(_b, c * 4, h // 2, w // 2)

    return self.conv(x)
