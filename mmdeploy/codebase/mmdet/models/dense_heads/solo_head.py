# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List

import torch
from mmdet.models.layers import mask_matrix_nms
from mmdet.utils import OptConfigType
from torch import Tensor
from torch.nn import functional as F

from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.SOLOHead.predict_by_feat', backend='openvino')
def solohead__predict_by_feat__openvino(ctx,
                                        self,
                                        mlvl_mask_preds: List[Tensor],
                                        mlvl_cls_scores: List[Tensor],
                                        batch_img_metas: List[Dict],
                                        cfg: OptConfigType = None,
                                        **kwargs):
    """Rewrite `predict_by_feat` of `SOLOHead` for openvino backend."""

    def tensor_tail_pad(tensor_var: torch.Tensor,
                        pad_value: Any = 0,
                        pad_num: Any = 1) -> torch.Tensor:
        tensor_tail = tensor_var.new_full((pad_num, *tensor_var.shape[1:]),
                                          pad_value)

        return torch.cat((tensor_var, tensor_tail), 0)

    cfg = self.test_cfg
    mlvl_cls_scores = [
        item.permute(0, 2, 3, 1).view(item.size(0), -1, self.cls_out_channels)
        for item in mlvl_cls_scores
    ]

    lvl_strides = [
        torch.ones_like(mlvl_cls_scores[lvl][0, :, 0]) * self.strides[lvl]
        for lvl in range(len(mlvl_cls_scores))
    ]
    lvl_strides = torch.cat(lvl_strides, 0)
    assert len(mlvl_mask_preds) == len(mlvl_cls_scores)
    batch_mlvl_cls_scores = torch.cat(mlvl_cls_scores, dim=1)
    batch_mlvl_mask_preds = torch.cat(mlvl_mask_preds, dim=1)
    featmap_size = batch_mlvl_mask_preds.size()[-2:]

    batch_dets, batch_labels, batch_masks = [], [], []
    for cls_scores, mask_preds, img_meta in zip(batch_mlvl_cls_scores,
                                                batch_mlvl_mask_preds,
                                                batch_img_metas):
        assert len(cls_scores) == len(mask_preds)

        score_mask = (cls_scores > cfg.score_thr)
        cls_scores = cls_scores[score_mask]

        inds = score_mask.nonzero()
        cls_labels = inds[:, 1]

        # Filter the mask mask with an area is smaller than
        strides = lvl_strides[inds[:, 0]]
        mask_preds = mask_preds[inds[:, 0]]

        masks = mask_preds > cfg.mask_thr
        sum_masks = masks.sum((1, 2)).float()
        keep = sum_masks > strides

        masks = masks[keep]
        mask_preds = mask_preds[keep]
        sum_masks = sum_masks[keep]
        cls_scores = cls_scores[keep]
        cls_labels = cls_labels[keep]

        # pad for zero-dim
        masks = tensor_tail_pad(masks)
        mask_preds = tensor_tail_pad(mask_preds)
        sum_masks = tensor_tail_pad(sum_masks, pad_value=1e-6)
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

        mask_preds = mask_preds[keep_inds]

        mmdet_params = get_post_processing_params(ctx.cfg)
        export_postprocess_mask = mmdet_params.get('export_postprocess_mask',
                                                   True)
        export_for_old_openvino_api = mmdet_params.get(
            'export_for_old_openvino_api', True)

        if export_for_old_openvino_api and cfg.max_per_img > 0:
            # set fixed malloc space
            max_per_img = cfg.max_per_img
            keep_len = mask_preds.size(0)
            _mask_preds = mask_preds.new_zeros(
                (max_per_img, *mask_preds.shape[1:]))
            _labels = cls_labels.new_zeros((max_per_img, ))
            _scores = cls_scores.new_zeros((max_per_img, ))

            # set dynamic preds to fixed malloc space
            _mask_preds[:keep_len, :, :] = mask_preds
            _labels[:keep_len] = labels
            _scores[:keep_len] = scores

            # change value name
            mask_preds = _mask_preds
            labels = _labels
            scores = _scores
        elif export_for_old_openvino_api and cfg.max_per_img <= 0:
            raise TypeError
        else:
            # openvino 2.0api can deal dynamic output, and here is for 1 batch
            scores = tensor_tail_pad(scores)
            labels = tensor_tail_pad(labels)
            mask_preds = tensor_tail_pad(mask_preds)

        h, w = img_meta['img_shape'][:2]
        if export_postprocess_mask:
            upsampled_size = (featmap_size[0] * 4, featmap_size[1] * 4)
            mask_preds = F.interpolate(
                mask_preds.unsqueeze(0), size=upsampled_size,
                mode='bilinear').squeeze(0)

        # add full image bbox
        bboxes = scores.new_zeros(scores.size(0), 2)
        bboxes = torch.cat([
            bboxes,
            bboxes.new_full((bboxes.size(0), 1), w),
            bboxes.new_full((bboxes.size(0), 1), h)
        ],
                           dim=1)
        dets = torch.cat((bboxes, scores[:, None]), dim=1)

        # add batch list
        batch_dets.append(dets)
        batch_labels.append(labels)
        batch_masks.append(mask_preds)

    batch_dets = torch.stack(batch_dets, 0)
    batch_labels = torch.stack(batch_labels, 0)
    batch_masks = torch.stack(batch_masks, 0)

    return batch_dets, batch_labels, batch_masks
