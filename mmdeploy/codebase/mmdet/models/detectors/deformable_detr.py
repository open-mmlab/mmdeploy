# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
from mmdet.structures.det_data_sample import OptSampleList
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.deformable_detr.DeformableDETR.pre_transformer')
def deformable_detr__pre_transformer(
        self,
        mlvl_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
    """Rewrite `pre_transformer` for default backend.

    Support exporting without masks for padding info.

    Args:
        mlvl_feats (tuple[Tensor]): Multi-level features that may have
            different resolutions, output from neck. Each feature has
            shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
        batch_data_samples (list[:obj:`DetDataSample`], optional): The
            batch data samples. It usually includes information such
            as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            Defaults to None.

    Returns:
        tuple[dict]: The first dict contains the inputs of encoder and the
        second dict contains the inputs of decoder.
    """
    # construct binary masks for the transformer.
    assert batch_data_samples is not None
    batch_size = mlvl_feats[0].shape[0]
    device = mlvl_feats[0].device
    mlvl_masks = []
    mlvl_pos_embeds = []
    for feat in mlvl_feats:
        mlvl_masks.append(None)
        shape_info = dict(
            B=batch_size, H=feat.shape[2], W=feat.shape[3], device=device)
        mlvl_pos_embeds.append(
            self.positional_encoding(mask=None, **shape_info))

    feat_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
        batch_size, c, h, w = feat.shape
        spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
        # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
        feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
        lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
        # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
        feat_flatten.append(feat)
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        spatial_shapes.append(spatial_shape)

    # (bs, num_feat_points, dim)
    feat_flatten = torch.cat(feat_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
    mask_flatten = None

    # (num_level, 2)
    spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
    level_start_index = torch.cat((
        spatial_shapes.new_zeros((1, )),  # (num_level)
        spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.ones(
        batch_size, len(mlvl_feats), 2, device=device)  # (bs, num_level, 2)

    encoder_inputs_dict = dict(
        feat=feat_flatten,
        feat_mask=mask_flatten,
        feat_pos=lvl_pos_embed_flatten,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios)
    decoder_inputs_dict = dict(
        memory_mask=mask_flatten,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios)
    return encoder_inputs_dict, decoder_inputs_dict


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.deformable_detr.'
    'DeformableDETR.gen_encoder_output_proposals')
def deformable_detr__gen_encoder_output_proposals(
        self, memory: Tensor, memory_mask: Tensor,
        spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
    """Rewrite `gen_encoder_output_proposals` for default backend.

    Support exporting with `memory_mask=None`.

    Args:
        memory (Tensor): The output embeddings of the Transformer encoder,
            has shape (bs, num_feat_points, dim).
        memory_mask (Tensor): ByteTensor, the padding mask of the memory,
            has shape (bs, num_feat_points).
        spatial_shapes (Tensor): Spatial shapes of features in all levels,
            has shape (num_levels, 2), last dimension represents (h, w).

    Returns:
        tuple: A tuple of transformed memory and proposals.

        - output_memory (Tensor): The transformed memory for obtaining
          top-k proposals, has shape (bs, num_feat_points, dim).
        - output_proposals (Tensor): The inverse-normalized proposal, has
          shape (batch_size, num_keys, 4) with the last dimension arranged
          as (cx, cy, w, h).
    """
    assert memory_mask is None, 'only support `memory_mask=None`'
    bs = memory.size(0)
    proposals = []
    for lvl, HW in enumerate(spatial_shapes):
        H, W = HW
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                0, H - 1, H, dtype=torch.float32, device=memory.device),
            torch.linspace(
                0, W - 1, W, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(bs, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
        proposals.append(proposal)
    output_proposals = torch.cat(proposals, 1)
    # do not use `all` to make it exportable to onnx
    output_proposals_valid = ((output_proposals > 0.01) &
                              (output_proposals < 0.99)).sum(
                                  -1,
                                  keepdim=True) == output_proposals.shape[-1]
    # inverse_sigmoid
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid,
                                                    float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(~output_proposals_valid,
                                              float(0))
    output_memory = self.memory_trans_fc(output_memory)
    output_memory = self.memory_trans_norm(output_memory)
    # [bs, sum(hw), 2]
    return output_memory, output_proposals
