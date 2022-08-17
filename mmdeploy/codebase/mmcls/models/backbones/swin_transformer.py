# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.core.rewriters.rewriter_utils import LibVersionChecker


@FUNCTION_REWRITER.register_rewriter(
    func_name=  # noqa: E251
    'mmcls.models.utils.attention.WindowMSA.forward',
    extra_checkers=LibVersionChecker('mmcls', min_version='0.21.0'))
def window_msa__forward__default(ctx, self, x, mask=None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                              C // self.num_heads).permute(2, 0, 3, 1, 4)
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
        attn = attn.view(B_ // nW, nW, self.num_heads, N,
                         N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)

    means = torch.mean(attn, self.softmax.dim, keepdim=True)[0]
    attn_exp = torch.exp(attn - means)
    attn_exp_sum = torch.sum(attn_exp, self.softmax.dim, keepdim=True)
    attn = attn_exp / attn_exp_sum
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


@FUNCTION_REWRITER.register_rewriter(
    func_name=  # noqa: E251
    'mmcls.models.utils.ShiftWindowMSA.forward',
    extra_checkers=LibVersionChecker('mmcls', min_version='0.21.0'))
def shift_window_msa__forward__default(ctx, self, query, hw_shape):
    # return ctx.origin_func(self, query, hw_shape)
    B, L, C = query.shape
    H, W = hw_shape
    assert L == H * W, f"The query length {L} doesn't match the input "\
        f'shape ({H}, {W}).'
    query = query.view(B, H, W, C)

    window_size = self.window_size
    shift_size = self.shift_size

    if min(H, W) == window_size:
        # If not pad small feature map, avoid shifting when the window size
        # is equal to the size of feature map. It's to align with the
        # behavior of the original implementation.
        shift_size = shift_size if self.pad_small_map else 0
    elif min(H, W) < window_size:
        # In the original implementation, the window size will be shrunk
        # to the size of feature map. The behavior is different with
        # swin-transformer for downstream tasks. To support dynamic input
        # shape, we don't allow this feature.
        assert self.pad_small_map, \
            f'The input shape ({H}, {W}) is smaller than the window ' \
            f'size ({window_size}). Please set `pad_small_map=True`, or ' \
            'decrease the `window_size`.'

    # pad feature maps to multiples of window size
    query = query.permute(0, 3, 1, 2).contiguous()
    # query = torch.nn.ZeroPad2d([0, self.window_size, 0, self.window_size])(
    #     query)
    query = torch.cat([query, query.new_zeros(B, C, H, window_size)], dim=-1)
    query = torch.cat(
        [query, query.new_zeros(B, C, window_size, query.shape[-1])], dim=-2)
    slice_h = (H + window_size - 1) // window_size * window_size
    slice_w = (W + window_size - 1) // window_size * window_size
    query = query[:, :, :slice_h, :slice_w]
    query = query.permute(0, 2, 3, 1).contiguous()
    H_pad, W_pad = query.shape[1], query.shape[2]

    # cyclic shift
    if shift_size > 0:
        query = torch.roll(
            query, shifts=(-shift_size, -shift_size), dims=(1, 2))

    attn_mask = self.get_attn_mask((H_pad, W_pad),
                                   window_size=window_size,
                                   shift_size=shift_size,
                                   device=query.device)

    # nW*B, window_size, window_size, C
    query_windows = self.window_partition(query, window_size)
    # nW*B, window_size*window_size, C
    query_windows = query_windows.view(-1, window_size**2, C)

    # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
    attn_windows = self.w_msa(query_windows, mask=attn_mask)

    # merge windows
    attn_windows = attn_windows.view(-1, window_size, window_size, C)

    # B H' W' C
    shifted_x = self.window_reverse(attn_windows, H_pad, W_pad, window_size)
    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
    else:
        x = shifted_x

    if H != H_pad or W != W_pad:
        x = x[:, :H, :W, :].contiguous()

    x = x.view(B, H * W, C)

    x = self.drop(x)

    return x


@FUNCTION_REWRITER.register_rewriter(
    func_name=  # noqa: E251
    'mmcls.models.utils.ShiftWindowMSA.get_attn_mask',
    extra_checkers=LibVersionChecker('mmcls', min_version='0.21.0'))
def shift_window_msa__get_attn_mask__default(ctx,
                                             self,
                                             hw_shape,
                                             window_size,
                                             shift_size,
                                             device=None):
    if shift_size > 0:
        # calculate attention mask for SW-MSA
        w_mask = torch.cat([
            torch.zeros((hw_shape[1] - window_size),
                        dtype=torch.int64,
                        device=device),
            torch.full((window_size - shift_size, ), 1, device=device),
            torch.full((shift_size, ), 2, device=device)
        ])
        h_mask = torch.cat([
            torch.zeros((hw_shape[0] - window_size),
                        dtype=torch.int64,
                        device=device),
            torch.full((window_size - shift_size, ), 3, device=device),
            torch.full((shift_size, ), 6, device=device)
        ])

        img_mask = w_mask.unsqueeze(0) + h_mask.unsqueeze(1)
        img_mask = img_mask.unsqueeze(0)
        img_mask = img_mask.unsqueeze(-1)
        # nW, window_size, window_size, 1
        from mmcls.models.utils import ShiftWindowMSA
        mask_windows = ShiftWindowMSA.window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    else:
        attn_mask = None
    return attn_mask
