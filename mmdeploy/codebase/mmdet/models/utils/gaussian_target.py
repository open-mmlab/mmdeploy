# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.utils.gaussian_target.get_topk_from_heatmap')
def get_topk_from_heatmap__default(scores, k=20):
    """Get top k positions from heatmap.

    Replace view(batch, -1) with flatten
    """
    height, width = scores.size()[2:]
    topk_scores, topk_inds = torch.topk(scores.flatten(1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
