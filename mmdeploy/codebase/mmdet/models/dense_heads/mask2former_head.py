# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.dense_heads.mask2former_head.'
    'Mask2FormerHead.forward')
def mask2former_head__forward(self, x, batch_data_samples):
    """Rewrite `forward` for default backend.

    Args:
        x (list[Tensor]): Multi scale Features from the
            upstream network, each is a 4D-tensor.
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        tuple[list[Tensor]]: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                h, w).
    """
    batch_size = x[0].shape[0]
    mask_features, multi_scale_memorys = self.pixel_decoder(x)
    # multi_scale_memorys (from low resolution to high resolution)
    decoder_inputs = []
    decoder_positional_encodings = []
    for i in range(self.num_transformer_feat_level):
        decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
        # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
        decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
        level_embed = self.level_embed.weight[i].view(1, 1, -1)
        decoder_input = decoder_input + level_embed
        # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
        mask = decoder_input.new_zeros(
            (batch_size, ) + multi_scale_memorys[i].shape[-2:],
            dtype=torch.bool)
        decoder_positional_encoding = self.decoder_positional_encoding(mask)
        decoder_positional_encoding = decoder_positional_encoding.flatten(
            2).permute(0, 2, 1)
        decoder_inputs.append(decoder_input)
        decoder_positional_encodings.append(decoder_positional_encoding)
    # shape (num_queries, c) -> (batch_size, num_queries, c)
    query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
    query_embed = self.query_embed.weight.unsqueeze(0).repeat(
        (batch_size, 1, 1))

    cls_pred_list = []
    mask_pred_list = []
    cls_pred, mask_pred, attn_mask = self._forward_head(
        query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
    cls_pred_list.append(cls_pred)
    mask_pred_list.append(mask_pred)

    for i in range(self.num_transformer_decoder_layers):
        level_idx = i % self.num_transformer_feat_level
        # if a mask is all True(all background), then set it all False.

        # to avoid Nonzero, replace with following code
        # attn_mask[torch.where(
        #     attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        cond = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(2)
        attn_mask = attn_mask & cond

        # cross_attn + self_attn
        layer = self.transformer_decoder.layers[i]
        query_feat = layer(
            query=query_feat,
            key=decoder_inputs[level_idx],
            value=decoder_inputs[level_idx],
            query_pos=query_embed,
            key_pos=decoder_positional_encodings[level_idx],
            cross_attn_mask=attn_mask,
            query_key_padding_mask=None,
            # here we do not apply masking on padded region
            key_padding_mask=None)
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features,
            multi_scale_memorys[(i + 1) %
                                self.num_transformer_feat_level].shape[-2:])

        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

    return cls_pred_list, mask_pred_list
