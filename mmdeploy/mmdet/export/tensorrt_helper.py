from typing import Any, Optional

import torch
from torch import Tensor


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
