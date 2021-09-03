from typing import Union

import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.linear', backend='ncnn')
def linear_ncnn(
    ctx,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, torch.NoneType] = None,
):
    origin_func = ctx.origin_func

    dim = input.dim()

    if dim == 2:
        return origin_func(input, weight, bias)
    else:
        out = origin_func(input, weight)

        # permute
        out = out.transpose(1, dim - 1)

        # ncnn only support [c, h, w] and [c, 1, 1] broadcast
        out_shape = out.shape
        batch_size = out_shape[0]
        broad_cast_size = out_shape[1]
        out = out.reshape([batch_size, broad_cast_size, -1, 1])

        # add bias
        bias = bias.view([1, -1, 1, 1])
        out = out + bias

        # permute back
        out = out.reshape(out_shape)
        out = out.transpose(1, dim - 1)

        return out
