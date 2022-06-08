# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Union

import mmcv
import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.core.rewriters.rewriter_utils import LibVersionChecker
from mmdeploy.utils import Backend, load_config


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


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet.deploy.utils.clip_bboxes',
    backend='tensorrt',
    extra_checkers=LibVersionChecker('tensorrt', min_version='8'))
def clip_bboxes__trt8(ctx, x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor,
                      max_shape: Union[Tensor, Sequence[int]]):
    """Clip bboxes for onnx. From TensorRT 8 we can do the operators on the
    tensors directly.

    Args:
        ctx (ContextCaller): The context with additional information.
        x1 (Tensor): The x1 for bounding boxes.
        y1 (Tensor): The y1 for bounding boxes.
        x2 (Tensor): The x2 for bounding boxes.
        y2 (Tensor): The y2 for bounding boxes.
        max_shape (Tensor | Sequence[int]): The (H,W) of original image.
    Returns:
        tuple(Tensor): The clipped x1, y1, x2, y2.
    """
    assert len(max_shape) == 2, '`max_shape` should be [h, w]'
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
    x_shape = list(x.shape)
    pad_shape = x_shape[:pad_dim] + [pad_size] + x_shape[pad_dim + 1:]
    x_pad = x.new_zeros(pad_shape)
    if pad_value is not None:
        x_pad = x_pad + pad_value
    x = torch.cat([x, x_pad], dim=pad_dim)
    return x


def pad_with_value_if_necessary(x: Tensor,
                                pad_dim: int,
                                pad_size: int,
                                pad_value: Optional[Any] = None):
    """Pad a tensor with a value along some dim if necessary.

    Args:
        x (Tensor): Input tensor.
        pad_dim (int): Along which dim to pad.
        pad_size (int): To which size to pad.
        pad_value (Any): Filled value for padding. Defaults to `None`.

    Returns:
        Tensor: Padded tensor.
    """
    return __pad_with_value_if_necessary(
        x, pad_dim, pad_size=pad_size, pad_value=pad_value)


def __pad_with_value_if_necessary(x: Tensor,
                                  pad_dim: int,
                                  pad_size: int,
                                  pad_value: Optional[Any] = None):
    """Pad a tensor with a value along some dim, do nothing on default.

    Args:
        x (Tensor): Input tensor.
        pad_dim (int): Along which dim to pad.
        pad_size (int): To which size to pad.
        pad_value (Any): Filled value for padding. Defaults to `None`.

    Returns:
        Tensor: Padded tensor.
    """
    return x


@FUNCTION_REWRITER.register_rewriter(
    'mmdeploy.codebase.mmdet.deploy.utils.__pad_with_value_if_necessary',
    backend=Backend.TENSORRT.value)
def __pad_with_value_if_necessary__tensorrt(ctx,
                                            x: Tensor,
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
    return pad_with_value(x, pad_dim, pad_size=pad_size, pad_value=pad_value)
