# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.layers.positional_encoding.'
    'SinePositionalEncoding.forward')
def sine_positional_encoding_forward__default(self, mask: Tensor,
                                              **kwargs) -> Tensor:
    """Rewrite `forward` for default backend.
    `mask=None` for single image inference
    Args:
        mask (Tensor | None): ByteTensor mask. Non-zero values representing
            ignored positions, while zero values means valid positions
            for this image. Shape [bs, h, w].

    Returns:
        pos (Tensor): Returned position embedding with shape
            [bs, num_feats*2, h, w].
    """
    if mask is not None:
        B, H, W = mask.shape
        device = mask.device
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
    else:
        B, H, W, = kwargs['B'], kwargs['H'], kwargs['W']
        device = kwargs['device']
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device)
        x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device)
        y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)

    if self.normalize:
        y_embed = (y_embed + self.offset) / \
                  (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = (x_embed + self.offset) / \
                  (x_embed[:, :, -1:] + self.eps) * self.scale
    dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
    dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    # use `view` instead of `flatten` for dynamically exporting to ONNX

    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
        dim=4).view(B, H, W, -1)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
        dim=4).view(B, H, W, -1)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos
