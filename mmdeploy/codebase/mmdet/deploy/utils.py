# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Sequence, Tuple, Union

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


def __gather_topk(*inputs: Sequence[torch.Tensor],
                  inds: torch.Tensor,
                  batch_size: int,
                  is_batched: bool = True) -> Tuple[torch.Tensor]:
    """The default implementation of gather_topk."""
    if is_batched:
        batch_inds = torch.arange(batch_size, device=inds.device).unsqueeze(-1)
        outputs = [
            x[batch_inds, inds, ...] if x is not None else None for x in inputs
        ]
    else:
        prior_inds = inds.new_zeros((1, 1))
        outputs = [
            x[prior_inds, inds, ...] if x is not None else None for x in inputs
        ]

    return outputs


class TRTGatherTopk(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, inds: torch.Tensor):
        """Implement of gather topk."""
        batch_size = x.size(0)
        batch_inds = torch.arange(batch_size, device=inds.device).unsqueeze(-1)
        return x[batch_inds, inds, ...]

    @staticmethod
    def symbolic(g, x, inds):
        """symbolic of gather topk."""
        out = g.op('mmdeploy::GatherTopk', x, inds, outputs=1)

        return out


@FUNCTION_REWRITER.register_rewriter(
    'mmdeploy.codebase.mmdet.deploy.utils.__gather_topk',
    backend=Backend.TENSORRT.value)
def __gather_topk__trt(ctx,
                       *inputs: Sequence[torch.Tensor],
                       inds: torch.Tensor,
                       batch_size: int,
                       is_batched: bool = True) -> Tuple[torch.Tensor]:
    """TensorRT gather_topk."""
    _ = ctx
    if is_batched:
        index_shape = inds.shape
        index_dim = inds.dim()
        outputs = [None for _ in inputs]
        for i, x in enumerate(inputs):
            if x is None:
                continue
            out = TRTGatherTopk.apply(x, inds).to(x.dtype)
            out_shape = [*index_shape, *x.shape[index_dim:]]
            out = out.reshape(out_shape)
            outputs[i] = out
    else:
        prior_inds = inds.new_zeros((1, 1))
        outputs = [
            x[prior_inds, inds, ...] if x is not None else None for x in inputs
        ]

    return outputs


@FUNCTION_REWRITER.register_rewriter(
    'mmdeploy.codebase.mmdet.deploy.utils.__gather_topk',
    backend=Backend.COREML.value)
def __gather_topk__nonbatch(ctx,
                            *inputs: Sequence[torch.Tensor],
                            inds: torch.Tensor,
                            batch_size: int,
                            is_batched: bool = True) -> Tuple[torch.Tensor]:
    """Single batch gather_topk."""
    assert batch_size == 1
    inds = inds.squeeze(0)
    outputs = [x[:, inds, ...] if x is not None else None for x in inputs]

    return outputs


def gather_topk(*inputs: Sequence[torch.Tensor],
                inds: torch.Tensor,
                batch_size: int,
                is_batched: bool = True) -> Tuple[torch.Tensor]:
    """Gather topk of each tensor.

    Args:
        inputs (Sequence[torch.Tensor]): Tensors to be gathered.
        inds (torch.Tensor): Topk index.
        batch_size (int): batch_size.
        is_batched (bool): Inputs is batched or not.

    Returns:
        Tuple[torch.Tensor]: Gathered tensors.
    """
    import mmdeploy
    outputs = mmdeploy.codebase.mmdet.deploy.utils.__gather_topk(
        *inputs, inds=inds, batch_size=batch_size, is_batched=is_batched)

    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs
