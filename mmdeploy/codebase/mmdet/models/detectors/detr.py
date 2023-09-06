# Copyright (c) OpenMMLab. All rights reserved.

from mmdet.structures.det_data_sample import OptSampleList

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.detr.DETR.pre_transformer')
def detr__pre_transformer(self, img_feats, batch_data_samples: OptSampleList):
    """Rewrite `pre_transformer` for default backend.

    Support exporting without masks for padding info.

    Args:
        img_feats (Tuple[Tensor]): Tuple of features output from the neck,
            has shape (bs, c, h, w).
        batch_data_samples (List[:obj:`DetDataSample`]): The batch
            data samples. It usually includes information such as
            `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            Defaults to None.

    Returns:
        tuple[dict, dict]: The first dict contains the inputs of encoder
        and the second dict contains the inputs of decoder.
    """
    feat = img_feats[-1]  # NOTE img_feats contains only one feature.
    batch_size, feat_dim, h, w = feat.shape
    # construct binary masks which for the transformer.
    assert batch_data_samples is not None
    masks = None  # for single image inference
    # [batch_size, embed_dim, h, w]
    extra_kwargs = dict(B=batch_size, H=h, W=w, device=feat.device)
    pos_embed = self.positional_encoding(mask=masks, **extra_kwargs)

    # use `view` instead of `flatten` for dynamically exporting to ONNX
    # [bs, c, h, w] -> [bs, h*w, c]
    feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
    pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
    # [bs, h, w] -> [bs, h*w]

    # prepare transformer_inputs_dict
    encoder_inputs_dict = dict(feat=feat, feat_mask=masks, feat_pos=pos_embed)
    decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
    return encoder_inputs_dict, decoder_inputs_dict
