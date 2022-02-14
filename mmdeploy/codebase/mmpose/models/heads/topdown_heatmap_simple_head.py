# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.heads.TopdownHeatmapSimpleHead.inference_model')
def topdown_heatmap_simple_head__inference_model(ctx,
                                                 self,
                                                 x,
                                                 flip_pairs=None):
    """Rewrite `forward_test` of TopDown for default backend.

    Rewrite this function to run forward directly. And we don't need to
    transform result to np.ndarray.

    Args:
        x (torch.Tensor[N,K,H,W]): Input features.
        flip_pairs (None | list[tuple]):
            Pairs of keypoints which are mirrored.

    Returns:
        output_heatmap (torch.Tensor): Output heatmaps.
    """
    assert flip_pairs is None
    output = self.forward(x)
    return output
