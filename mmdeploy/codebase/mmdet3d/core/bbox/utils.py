# Copyright (c) OpenMMLab. All rights reserved.
import torch


def points_img2cam(points, cam2img_inverse):
    """Project points in image coordinates to camera coordinates.

    Rewrite this
    func for remove inverse op.
    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img_inverse (torch.Tensor): The inverse of camera intrinsic matrix.
        The shape can be [3, 3], [3, 4] or [4, 4].
    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img_inverse.shape[0] <= 4
    assert cam2img_inverse.shape[1] <= 4
    assert points.shape[2] == 3

    xys = points[..., :2]
    depths = points[..., 2].unsqueeze(-1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=-1)

    inv_pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    inv_pad_cam2img[:cam2img_inverse.shape[0], :cam2img_inverse.
                    shape[1]] = cam2img_inverse
    inv_pad_cam2img = inv_pad_cam2img.transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[1]
    homo_xys = torch.cat(
        [unnormed_xys, xys.new_ones((1, num_points, 1))], dim=-1)
    points3D = torch.matmul(homo_xys, inv_pad_cam2img)[..., :3]

    return points3D
