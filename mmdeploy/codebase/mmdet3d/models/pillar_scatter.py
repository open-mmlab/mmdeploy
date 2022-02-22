import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.middle_encoders.pillar_scatter.'
    'PointPillarsScatter.forward_batch',
    backend='tensorrt')
def pointpillarsscatter__forward(ctx, self, voxel_features, coors, batch_size):
    canvas = torch.zeros(
        self.in_channels,
        self.nx * self.ny,
        dtype=voxel_features.dtype,
        device=voxel_features.device)

    indices = coors[:, 2] * self.nx + coors[:, 3]
    indices = indices.long()
    voxels = voxel_features.t()
    # Now scatter the blob back to the canvas.
    canvas[:, indices] = voxels
    canvas[:, 0] = 0
    # Undo the column stacking to final 4-dim tensor
    canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
    return canvas
