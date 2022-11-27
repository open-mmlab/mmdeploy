from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.utils import OptConfigType
from mmdet.models.layers import mask_matrix_nms

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.codebase.mmdet.deploy import get_post_processing_params


def tensor_tail_pad(tensor_var: torch.Tensor, 
                    pad_value: Any =0, 
                    pad_num: int =1,
                    dim: int =0) -> torch.Tensor :
    if dim == 0:
        tensor_tail = tensor_var.new_full((pad_num, *tensor_var.shape[1:]), 
                                          pad_value)
    else:
        raise NotImplementedError
    
    return torch.cat((tensor_var, tensor_tail), dim=dim)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.SOLOHead._predict_by_feat_single')
def solo_head___predict_by_feat_single(ctx,
                                       self,
                                       cls_scores: Tensor,
                                       mask_preds: Tensor,
                                       img_meta: dict,
                                       cfg: OptConfigType = None) -> InstanceData:
    """Transform a single image's features extracted from the head into
    mask results.

    Args:
        cls_scores (Tensor): Classification score of all points
            in single image, has shape (num_points, num_classes).
        mask_preds (Tensor): Mask prediction of all points in
            single image, has shape (num_points, feat_h, feat_w).
        img_meta (dict): Meta information of corresponding image.
        cfg (dict, optional): Config used in test phase.
            Defaults to None.

    Returns:
        :obj:`InstanceData`: Processed results of single image.
            it usually contains following keys.

            - scores (Tensor): Classification scores, has shape
                (num_instance,).
            - labels (Tensor): Has shape (num_instances,).
            - masks (Tensor): Processed mask results, has
                shape (num_instances, h, w).
    """

    cfg = self.test_cfg if cfg is None else cfg
    assert len(cls_scores) == len(mask_preds)

    score_mask = (cls_scores > cfg.score_thr)
    cls_scores = cls_scores[score_mask]
    cls_scores = tensor_tail_pad(cls_scores)

    inds = score_mask.nonzero()
    cls_labels = inds[:, 1]
    cls_labels = tensor_tail_pad(cls_labels)

    # Filter the mask mask with an area is smaller than
    # stride of corresponding feature level
    lvl_interval = cls_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
    strides = cls_scores.new_ones(lvl_interval[-1])
    strides[:lvl_interval[0]] *= self.strides[0]
    for lvl in range(1, self.num_levels):
        strides[lvl_interval[lvl -
                                1]:lvl_interval[lvl]] *= self.strides[lvl]
    strides = strides[inds[:, 0]]
    strides = tensor_tail_pad(strides)
    mask_preds = mask_preds[inds[:, 0]]
    mask_preds = tensor_tail_pad(mask_preds)

    masks = mask_preds > cfg.mask_thr
    sum_masks = masks.sum((1, 2)).float()
    keep = sum_masks > strides

    masks = masks[keep]
    mask_preds = mask_preds[keep]
    sum_masks = sum_masks[keep]
    cls_scores = cls_scores[keep]
    cls_labels = cls_labels[keep]

    masks = tensor_tail_pad(masks)
    mask_preds = tensor_tail_pad(mask_preds)
    sum_masks = tensor_tail_pad(sum_masks, 1e-6)
    cls_scores = tensor_tail_pad(cls_scores)
    cls_labels = tensor_tail_pad(cls_labels)

    # maskness.
    mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
    cls_scores *= mask_scores

    scores, labels, _, keep_inds = mask_matrix_nms(
        masks,
        cls_labels,
        cls_scores,
        mask_area=sum_masks,
        nms_pre=cfg.nms_pre,
        max_num=cfg.max_per_img,
        kernel=cfg.kernel,
        sigma=cfg.sigma,
        filter_thr=cfg.filter_thr)
    scores = tensor_tail_pad(scores)
    labels = tensor_tail_pad(labels)
    
    mask_preds = mask_preds[keep_inds]
    mask_preds = tensor_tail_pad(mask_preds)

    mmdet_params = get_post_processing_params(ctx.cfg)
    export_postprocess_mask = mmdet_params.get('export_postprocess_mask', True)
    export_for_old_openvino_api = mmdet_params.get('export_for_old_openvino_api', True)

    if export_for_old_openvino_api and cfg.max_per_img > 0:
        max_per_img = cfg.max_per_img + 1
        keep_len = mask_preds.shape[0]
        _mask_preds = mask_preds.new_zeros((max_per_img, *mask_preds.shape[1:])) 
        _labels = labels.new_zeros((max_per_img, ))
        _scores = scores.new_zeros((max_per_img, ))
        _mask_preds[:keep_len, :, :] = mask_preds
        _labels[:keep_len] = labels
        _scores[:keep_len] = scores
        mask_preds = _mask_preds
        labels = _labels
        scores = _scores

    if export_postprocess_mask:
        h, w = img_meta['img_shape'][:2]
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(0), size=(h, w),
            mode='bilinear')
        mask_preds = mask_preds[0]
        masks = mask_preds > cfg.mask_thr

    results = InstanceData()
    results.masks = mask_preds
    results.labels = labels
    results.scores = scores
    # create an empty bbox in InstanceData to avoid bugs when
    # calculating metrics.
    results.bboxes = results.scores.new_zeros(scores.size(0), 4)

    return results