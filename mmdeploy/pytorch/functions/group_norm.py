# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.group_norm', backend='ncnn')
def group_norm__ncnn(
    ctx,
    input: torch.Tensor,
    num_groups: int,
    weight: Union[torch.Tensor, torch.NoneType] = None,
    bias: Union[torch.Tensor, torch.NoneType] = None,
    eps: float = 1e-05,
) -> torch.Tensor:
    """Rewrite `group_norm` for NCNN backend.

    InstanceNorm in ncnn require input with shape [C, H, W]. So we have to
    reshape the input tensor before it.
    """
    input_shape = input.shape
    batch_size = input_shape[0]
    # We cannot use input.reshape(batch_size, num_groups, -1, 1)
    # instead, or we will meet bug on ncnn Reshape ops.
    input_reshaped = input.reshape(batch_size, num_groups, -1)
    input_reshaped = input_reshaped.unsqueeze(3)
    # the weight_'s size is not the same as weight's size
    # we only use groupnorm to calculate instancenorm, but the
    # input parameters may not be the same, and need to transform.
    weight_ = torch.tensor([1.] * num_groups).type_as(input)
    bias_ = torch.tensor([0.] * num_groups).type_as(input)

    norm_reshaped = torch.nn.functional.instance_norm(
        input_reshaped, weight=weight_, bias=bias_, eps=eps)

    norm = norm_reshaped.reshape(*input_shape)
    if weight is None:
        weight = torch.tensor([1.]).type_as(input)
    if bias is None:
        bias = torch.tensor([0.]).type_as(input)
    weight = weight.reshape(1, -1, 1, 1)
    bias = bias.reshape(1, -1, 1, 1)

    return norm * weight + bias
