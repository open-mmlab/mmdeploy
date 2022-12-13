# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.middle_encoders.pillar_scatter.'
    'PointPillarsScatter.forward_batch', )
def pointpillarsscatter__forward(self, voxel_features, coors, batch_size=1):
    """Scatter features of single sample.

    Args:
        voxel_features (torch.Tensor): Voxel features from voxel encoder layer.
        coors (torch.Tensor): Coordinates of each voxel.
            The first column indicates the sample ID.
        batch_size (int): Number of samples in the current batch, batch_size=1
            by default.
    """
    canvas = torch.zeros(
        self.in_channels,
        self.nx * self.ny,
        dtype=voxel_features.dtype,
        device=voxel_features.device)

    indices = coors[:, 2] * self.nx + coors[:, 3]
    indices = indices.long()
    voxels = voxel_features.t()
    # Now scatter the blob back to the canvas.

    canvas.scatter_(
        dim=1, index=indices.expand(canvas.shape[0], -1), src=voxels)
    # Undo the column stacking to final 4-dim tensor
    canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
    return canvas
