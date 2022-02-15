import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.middle_encoders.pillar_scatter.'
    'PointPillarsScatter.forward_batch')
def forward(ctx, self, voxel_features, coors, batch_size):
    batch_canvas = []
    assert batch_size == 1
    for batch_itt in range(batch_size):
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        # Only include non-empty pillars
        voxel_num = (coors[:, 0] == batch_itt).shape[0]
        this_coors = coors[:voxel_num, :]
        indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
        indices = indices.type(torch.long)
        voxels = voxel_features[:voxel_num, :]
        voxels = voxels.t()

        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels

        # Append to a list for later stacking.
        batch_canvas.append(canvas)

    # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
    batch_canvas = torch.stack(batch_canvas, 0)

    # Undo the column stacking to final 4-dim tensor
    batch_canvas = batch_canvas.reshape(1, self.in_channels, self.ny, self.nx)

    return batch_canvas
