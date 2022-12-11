# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.core.anchor.anchor_3d_generator.AlignedAnchor3DRangeGenerator.'
    'anchors_single_range')
def alignedanchor3drangegenerator__anchors_single_range(
        ctx,
        self,
        feature_size,
        anchor_range,
        scale,
        sizes=[[3.9, 1.6, 1.56]],
        rotations=[0, 1.5707963],
        device='cuda'):
    """Generate anchors in a single range. Rewrite this func for default
    backend.

    Args:
        feature_size (list[float] | tuple[float]): Feature map size. It is
            either a list of a tuple of [D, H, W](in order of z, y, and x).
        anchor_range (torch.Tensor | list[float]): Range of anchors with
            shape [6]. The order is consistent with that of anchors, i.e.,
            (x_min, y_min, z_min, x_max, y_max, z_max).
        scale (float | int): The scale factor of anchors.
        sizes (list[list] | np.ndarray | torch.Tensor, optional):
            Anchor size with shape [N, 3], in order of x, y, z.
            Defaults to [[3.9, 1.6, 1.56]].
        rotations (list[float] | np.ndarray | torch.Tensor, optional):
            Rotations of anchors in a single feature grid.
            Defaults to [0, 1.5707963].
        device (str, optional): Devices that the anchors will be put on.
            Defaults to 'cuda'.

    Returns:
        torch.Tensor: Anchors with shape
            [*feature_size, num_sizes, num_rots, 7].
    """
    if len(feature_size) == 2:
        feature_size = [1, feature_size[0], feature_size[1]]
    anchor_range = torch.tensor(anchor_range, device=device)
    z_centers = torch.arange(feature_size[0], device=device)
    z_centers = z_centers.to(anchor_range.dtype)
    y_centers = torch.arange(feature_size[1], device=device)
    y_centers = y_centers.to(anchor_range.dtype)
    x_centers = torch.arange(feature_size[2], device=device)
    x_centers = x_centers.to(anchor_range.dtype)

    # shift the anchor center
    if not self.align_corner:
        z_centers += 0.5
        y_centers += 0.5
        x_centers += 0.5

    z_centers = z_centers / feature_size[0] * (
        anchor_range[5] - anchor_range[2]) + anchor_range[2]
    y_centers = y_centers / feature_size[1] * (
        anchor_range[4] - anchor_range[1]) + anchor_range[1]
    x_centers = x_centers / feature_size[2] * (
        anchor_range[3] - anchor_range[0]) + anchor_range[0]

    sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
    rotations = torch.tensor(rotations, device=device)

    # torch.meshgrid default behavior is 'id', np's default is 'xy'
    rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)

    # torch.meshgrid returns a tuple rather than list
    rets = list(rets)
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

    sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = sizes.repeat(tile_size_shape)
    rets.insert(3, sizes)

    ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])

    if len(self.custom_values) > 0:
        custom_ndim = len(self.custom_values)
        custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
        # TODO: check the support of custom values
        # custom[:] = self.custom_values
        ret = torch.cat([ret, custom], dim=-1)
    return ret
