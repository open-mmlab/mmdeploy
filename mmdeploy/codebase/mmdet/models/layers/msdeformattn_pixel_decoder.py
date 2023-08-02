# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.layers.msdeformattn_pixel_decoder.'
    'MSDeformAttnPixelDecoder.forward')
def msdeform_attn_pixel_decoder__forward(self, feats: List[torch.Tensor]):
    """Rewrite `forward` for default backend.
    Args:
        feats (list[Tensor]): Feature maps of each level. Each has
            shape of (batch_size, c, h, w).

    Returns:
        tuple: A tuple containing the following:

            - mask_feature (Tensor): shape (batch_size, c, h, w).
            - multi_scale_features (list[Tensor]): Multi scale \
                    features, each in shape (batch_size, c, h, w).
    """
    # generate padding mask for each level, for each image
    batch_size = feats[0].shape[0]
    encoder_input_list = []
    padding_mask_list = []
    level_positional_encoding_list = []
    spatial_shapes = []
    reference_points_list = []
    for i in range(self.num_encoder_levels):
        level_idx = self.num_input_levels - i - 1
        feat = feats[level_idx]
        feat_projected = self.input_convs[i](feat)
        feat_hw = torch._shape_as_tensor(feat)[2:]

        # no padding
        padding_mask_resized = feat.new_zeros(
            (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
        pos_embed = self.postional_encoding(padding_mask_resized)
        level_embed = self.level_encoding.weight[i]
        level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
        # (h_i * w_i, 2)
        reference_points = self.point_generator.single_level_grid_priors(
            feat.shape[-2:], level_idx, device=feat.device)
        # normalize
        feat_wh = feat_hw.unsqueeze(0).flip(dims=[0, 1])
        factor = feat_wh * self.strides[level_idx]
        reference_points = reference_points / factor

        # shape (batch_size, c, h_i, w_i) -> (h_i * w_i, batch_size, c)
        feat_projected = feat_projected.flatten(2).permute(0, 2, 1)
        level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)
        padding_mask_resized = padding_mask_resized.flatten(1)

        encoder_input_list.append(feat_projected)
        padding_mask_list.append(padding_mask_resized)
        level_positional_encoding_list.append(level_pos_embed)
        spatial_shapes.append(feat_hw)
        reference_points_list.append(reference_points)
    # shape (batch_size, total_num_queries),
    # total_num_queries=sum([., h_i * w_i,.])
    padding_masks = torch.cat(padding_mask_list, dim=1)
    # shape (total_num_queries, batch_size, c)
    encoder_inputs = torch.cat(encoder_input_list, dim=1)
    level_positional_encodings = torch.cat(
        level_positional_encoding_list, dim=1)
    # shape (num_encoder_levels, 2), from low
    # resolution to high resolution
    spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)

    # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
    # keep last index
    level_start_index = torch.cat(
        [spatial_shapes.new_zeros(1),
         spatial_shapes.prod(1).cumsum(0)]).to(torch.long)
    reference_points = torch.cat(reference_points_list, dim=0)
    reference_points = reference_points[None, :,
                                        None].repeat(batch_size, 1,
                                                     self.num_encoder_levels,
                                                     1)
    valid_radios = reference_points.new_ones(
        (batch_size, self.num_encoder_levels, 2))
    # shape (num_total_queries, batch_size, c)
    memory = self.encoder(
        query=encoder_inputs,
        query_pos=level_positional_encodings,
        key_padding_mask=padding_masks,
        spatial_shapes=spatial_shapes,
        reference_points=reference_points,
        level_start_index=level_start_index[:-1],
        valid_ratios=valid_radios)
    # (batch_size, c, num_total_queries)
    memory = memory.permute(0, 2, 1)

    # from low resolution to high resolution
    # num_queries_per_level = [e[0] * e[1] for e in spatial_shapes]
    # outs = torch.split(memory, num_queries_per_level, dim=-1)
    outs = []
    for i in range(self.num_encoder_levels):
        outs.append(memory[:, :,
                           level_start_index[i]:level_start_index[i + 1]])

    outs = [
        x.reshape(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1])
        for i, x in enumerate(outs)
    ]

    for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                   -1):
        x = feats[i]
        cur_feat = self.lateral_convs[i](x)
        y = cur_feat + F.interpolate(
            outs[-1],
            size=cur_feat.shape[-2:],
            mode='bilinear',
            align_corners=False)
        y = self.output_convs[i](y)
        outs.append(y)
    multi_scale_features = outs[:self.num_outs]

    mask_feature = self.mask_feature(outs[-1])
    return mask_feature, multi_scale_features
