# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER, SYMBOLIC_REWRITER


@SYMBOLIC_REWRITER.register_symbolic(
    'mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction')
def ms_deform_attn_default(
    g,
    value,
    value_spatial_shapes,
    value_level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step=64,
):
    """Rewrite msda symbolic function for all backend."""
    return g.op(
        'mmdeploy::MMCVMultiScaleDeformableAttention',
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step_i=im2col_step,
    )


@FUNCTION_REWRITER.register_rewriter(
    'mmcv.ops.multi_scale_deform_attn.multi_scale_deformable_attn_pytorch')
def multi_scale_deformable_attn_pytorch_default(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    indices = torch.cat((value_spatial_shapes.new_zeros(1),
                         value_spatial_shapes.prod(1).cumsum(0)))
    # avoid split with dynamic split_sizes
    # value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
    #                          dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for i in range(num_levels):
        H_, W_ = value_spatial_shapes[i]
        value_l_ = value[:, indices[i]:indices[i + 1], :, :]
        value_l_ = value_l_.flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          i].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()
