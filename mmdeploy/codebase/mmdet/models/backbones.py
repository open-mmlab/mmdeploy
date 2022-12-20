# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import get_common_config


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
    assert h % 2 == 0 and w % 2 == 0, 'focus for yolox needs even feature' \
        f'height and width, got {(h, w)}.'

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


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.backbones.swin.WindowMSA.forward',
    backend='tensorrt')
def windowmsa__forward__tensorrt(ctx, self, x, mask=None):
    """Rewrite forward function of WindowMSA class for TensorRT.

    1. replace Gather operation of qkv with split.
    2. replace SoftMax operation with a workaround done by PyTorch.

    Args:
        x (tensor): input features with shape of (num_windows*B, N, C)
        mask (tensor | None, Optional): mask with shape of (num_windows,
            Wh*Ww, Wh*Ww), value should be between (-inf, 0].
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                              -1).permute(2, 0, 3, 1, 4).contiguous()

    # replace the gather operation with the split
    q, k, v = [i.squeeze(0) for i in torch.split(qkv, 1, 0)]

    q = q * self.scale

    attn = (q @ k.transpose(-2, -1))

    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(
        2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(-1, nW, self.num_heads, N,
                         N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)

    # replace softmax with a workaround
    # weird bug from TensorRT. softmax cannot be used here for fp32 and it
    # can be used in fp16, but softmax fp16 performance is not as good as
    # exp and log_softmax. Besides, only the UT of exp and log_softmax passed.
    fp16_mode = get_common_config(ctx.cfg).get('fp16_mode', False)
    if fp16_mode:
        attn = torch.exp(torch.log_softmax(attn, dim=self.softmax.dim))
    else:
        means = torch.mean(attn, self.softmax.dim, keepdim=True)[0]
        attn_exp = torch.exp(attn - means)
        attn_exp_sum = torch.sum(attn_exp, self.softmax.dim, keepdim=True)
        attn = attn_exp / attn_exp_sum

    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.backbones.swin.ShiftWindowMSA.window_reverse',
    backend='tensorrt')
def shift_window_msa__window_reverse__tensorrt(ctx, self, windows, H, W):
    """Rewrite window_reverse function of ShiftWindowMSA class for TensorRT.
    For TensorRT, seems radical shape transformations are not allowed. Replace
    them with soft ones.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    window_size = self.window_size
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    # x = windows.view(B, H // window_size, W // window_size, window_size,
    #     window_size, -1)
    x = windows.view(B, -1, W, window_size, windows.shape[-1])
    x = x.view(B, x.shape[1], -1, window_size, window_size, x.shape[-1])
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, x.shape[-1])
    return x


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.backbones.swin.ShiftWindowMSA.window_partition',
    backend='tensorrt')
def shift_window_msa__window_partition__tensorrt(ctx, self, x):
    """Rewrite window_partition function of ShiftWindowMSA class for TensorRT.
    For TensorRT, seems radical shape transformations are not allowed. Replace
    them with soft ones.

    Args:
        x: (B, H, W, C)
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    window_size = self.window_size
    x = x.view(B, H, -1, window_size, C)
    x = x.view(B, -1, window_size, x.shape[-3], window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.backbones.swin.ShiftWindowMSA.forward')
def shift_window_msa__forward__default(ctx, self, query, hw_shape):
    """Rewrite forward function of ShiftWindowMSA class.

    1. replace dynamic padding with static padding and dynamic slice.
    2. always do slice `x = x[:, :H, :W, :].contiguous()` for stability.
    """
    B, L, C = query.shape
    H, W = hw_shape
    assert L == H * W, 'input feature has wrong size'
    query = query.view(B, H, W, C)

    # pad feature maps to multiples of window size
    query = query.permute(0, 3, 1, 2).contiguous()
    # query = torch.nn.ZeroPad2d([0, self.window_size, 0, self.window_size])(
    #     query)
    query = torch.cat(
        [query, query.new_zeros(B, C, H, self.window_size)], dim=-1)
    query = torch.cat(
        [query,
         query.new_zeros(B, C, self.window_size, query.shape[-1])],
        dim=-2)
    slice_h = (H + self.window_size - 1) // self.window_size * self.window_size
    slice_w = (W + self.window_size - 1) // self.window_size * self.window_size
    query = query[:, :, :slice_h, :slice_w]
    query = query.permute(0, 2, 3, 1).contiguous()
    H_pad, W_pad = query.shape[1], query.shape[2]

    # cyclic shift
    if self.shift_size > 0:
        shifted_query = torch.roll(
            query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # calculate attention mask for SW-MSA
        w_mask = torch.cat([
            shifted_query.new_zeros(W_pad - self.window_size),
            shifted_query.new_full((self.window_size - self.shift_size, ), 1),
            shifted_query.new_full((self.shift_size, ), 2)
        ])
        h_mask = torch.cat([
            shifted_query.new_zeros(H_pad - self.window_size),
            shifted_query.new_full((self.window_size - self.shift_size, ), 3),
            shifted_query.new_full((self.shift_size, ), 6)
        ])

        img_mask = w_mask.unsqueeze(0) + h_mask.unsqueeze(1)
        img_mask = img_mask.unsqueeze(0)
        img_mask = img_mask.unsqueeze(-1)

        # nW, window_size, window_size, 1
        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-100.0)).masked_fill(
                                              attn_mask == 0, float(0.0))
    else:
        shifted_query = query
        attn_mask = None

    # nW*B, window_size, window_size, C
    query_windows = self.window_partition(shifted_query)
    # nW*B, window_size*window_size, C
    query_windows = query_windows.view(-1, self.window_size**2, C)

    # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
    attn_windows = self.w_msa(query_windows, mask=attn_mask)

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

    # B H' W' C
    shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(
            shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x

    x = x[:, :H, :W, :].contiguous()

    x = x.view(B, H * W, C)

    x = self.drop(x)
    return x
