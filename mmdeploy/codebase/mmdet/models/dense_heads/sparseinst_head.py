# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.utils import aligned_bilinear
from mmdet.structures import OptSampleList, SampleList
from mmengine.config import ConfigDict
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([2, 3]) /
                     (mask_pred_.sum([2, 3]) + 1e-6))


@FUNCTION_REWRITER.register_rewriter(
    'projects.SparseInst.sparseinst.SparseInst.predict')
def sparseinst__predict(
    self,
    batch_inputs: Tensor,
    batch_data_samples: List[dict],
    rescale: bool = False,
):
    """Rewrite `predict` of `SparseInst` for default backend."""
    max_shape = batch_inputs.shape[-2:]
    x = self.extract_feat(batch_inputs)
    output = self.decoder(x)
    
    pred_scores = output['pred_logits'].sigmoid()
    pred_masks = output['pred_masks'].sigmoid()
    pred_objectness = output['pred_scores'].sigmoid()
    pred_scores = torch.sqrt(pred_scores * pred_objectness)

    # max/argmax
    scores, labels = pred_scores.max(dim=-1)
    # cls threshold
    keep = scores > self.cls_threshold
    scores = scores.where(keep, scores.new_zeros(1))
    labels = labels.where(keep, labels.new_zeros(1))
    keep = keep.unsqueeze(-1).unsqueeze(-1).expand_as(pred_masks)
    pred_masks = pred_masks.where(keep, pred_masks.new_zeros(1))

    img_meta = batch_data_samples[0].metainfo
    # rescoring mask using maskness
    scores = rescoring_mask(scores,
                            pred_masks > self.mask_threshold,
                            pred_masks)
    h, w = img_meta['img_shape'][:2]
    pred_masks = F.interpolate(pred_masks,
                               size=max_shape,
                               mode='bilinear',
                               align_corners=False)[:, :, :h, :w]
    
    bboxes = torch.zeros(scores.shape[0], scores.shape[1], 4)
    dets = torch.cat([bboxes, scores.unsqueeze(-1)], dim=-1)
    masks = (pred_masks > self.mask_threshold).float()

    return dets, labels, masks
