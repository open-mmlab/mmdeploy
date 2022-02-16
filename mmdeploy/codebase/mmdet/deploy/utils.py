# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Union

import mmcv
import torch
from torch import Tensor

from mmdeploy.utils import load_config


def get_post_processing_params(deploy_cfg: Union[str, mmcv.Config]):
    """Get mmdet post-processing parameters from config.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        dict: A dict of parameters for mmdet.
    """
    deploy_cfg = load_config(deploy_cfg)[0]
    codebase_key = 'codebase_config'
    assert codebase_key in deploy_cfg
    codebase_config = deploy_cfg[codebase_key]
    post_params = codebase_config.get('post_processing', None)
    assert post_params is not None, 'Failed to get `post_processing`.'
    return post_params


def clip_bboxes(x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor,
                max_shape: Union[Tensor, Sequence[int]]):
    """Clip bboxes for onnx.

    Since torch.clamp cannot have dynamic `min` and `max`, we scale the
      boxes by 1/max_shape and clamp in the range [0, 1] if necessary.

    Args:
        x1 (Tensor): The x1 for bounding boxes.
        y1 (Tensor): The y1 for bounding boxes.
        x2 (Tensor): The x2 for bounding boxes.
        y2 (Tensor): The y2 for bounding boxes.
        max_shape (Tensor | Sequence[int]): The (H,W) of original image.
    Returns:
        tuple(Tensor): The clipped x1, y1, x2, y2.
    """
    assert len(max_shape) == 2, '`max_shape` should be [h, w]'
    if isinstance(max_shape, torch.Tensor):
        # scale by 1/max_shape
        x1 = x1 / max_shape[1]
        y1 = y1 / max_shape[0]
        x2 = x2 / max_shape[1]
        y2 = y2 / max_shape[0]

        # clamp [0, 1]
        x1 = torch.clamp(x1, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        x2 = torch.clamp(x2, 0, 1)
        y2 = torch.clamp(y2, 0, 1)

        # scale back
        x1 = x1 * max_shape[1]
        y1 = y1 * max_shape[0]
        x2 = x2 * max_shape[1]
        y2 = y2 * max_shape[0]
    else:
        x1 = torch.clamp(x1, 0, max_shape[1])
        y1 = torch.clamp(y1, 0, max_shape[0])
        x2 = torch.clamp(x2, 0, max_shape[1])
        y2 = torch.clamp(y2, 0, max_shape[0])
    return x1, y1, x2, y2


def pad_with_value(x: Tensor,
                   pad_dim: int,
                   pad_size: int,
                   pad_value: Optional[Any] = None):
    """Pad a tensor with a value along some dim.

    Args:
        x (Tensor): Input tensor.
        pad_dim (int): Along which dim to pad.
        pad_size (int): To which size to pad.
        pad_value (Any): Filled value for padding. Defaults to `None`.

    Returns:
        Tensor: Padded tensor.
    """
    num_dims = len(x.shape)
    pad_slice = (slice(None, None, None), ) * num_dims
    pad_slice = pad_slice[:pad_dim] + (slice(0, 1,
                                             1), ) + pad_slice[pad_dim + 1:]
    repeat_size = [1] * num_dims
    repeat_size[pad_dim] = pad_size

    x_pad = x.__getitem__(pad_slice)
    if pad_value is not None:
        x_pad = x_pad * 0 + pad_value

    x_pad = x_pad.repeat(*repeat_size)
    x = torch.cat([x, x_pad], dim=pad_dim)
    return x
