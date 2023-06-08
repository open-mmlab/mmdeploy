# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.GuidedAnchorHead._get_guided_anchors_single')
def guided_anchor_head__get_guided_anchors_single(ctx,
                                                  self,
                                                  squares,
                                                  shape_pred,
                                                  loc_pred,
                                                  use_loc_filter=False):
    """Get guided anchors and loc masks for a single level.

    Args:
        square (tensor): Squares of a single level.
        shape_pred (tensor): Shape predictions of a single level.
        loc_pred (tensor): Loc predictions of a single level.
        use_loc_filter (list[tensor]): Use loc filter or not.
    Returns:
        tuple: guided anchors
    """
    # calculate location filtering mask
    # calculate guided anchors
    anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(-1,
                                                                  2).detach()
    bbox_deltas = anchor_deltas.new_full(squares.size(), 0)
    bbox_deltas[:, 2:] = anchor_deltas
    guided_anchors = self.anchor_coder.decode(
        squares, bbox_deltas, wh_ratio_clip=1e-6)
    return guided_anchors, None
