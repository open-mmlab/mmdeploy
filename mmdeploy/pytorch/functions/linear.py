# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch

from mmdeploy.core import FUNCTION_REWRITER


class GemmOp(torch.autograd.Function):
    """Create onnx::Gemm op."""

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        out = input @ weight.transpose(0, 1)
        if bias is not None:
            out += bias
        return out

    @staticmethod
    def symbolic(g, input, weight, bias=None):
        input.setDebugName('A')
        weight.setDebugName('B')
        args = ['Gemm', input, weight]
        if bias is not None:
            bias.setDebugName('C')
            args.append(bias)
        return g.op(*args, alpha_f=1.0, beta_f=1.0, transA_i=0, transB_i=1)


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.linear', backend='ncnn')
def linear__ncnn(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[Union[torch.Tensor, torch.NoneType]] = None,
):
    """Rewrite `linear` for ncnn backend.

    The broadcast rules are different between ncnn and PyTorch. This function
    add extra reshape and transpose to support linear operation of different
    input shape.
    """
    ctx = FUNCTION_REWRITER.get_context()
    origin_func = ctx.origin_func
    dim = input.dim()

    if dim == 2 or dim == 3 and input.shape[0] == 1:
        # export nn.linear to Gemm op in onnx
        return GemmOp.apply(input, weight, bias)
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
        if bias is not None:
            bias = bias.view([1, -1, 1, 1])
            out = out + bias

        # permute back
        # the last dim should be -1 to support dynamic shape
        out = out.reshape(out_shape[:-1] + (-1, ))
        out = out.transpose(1, dim - 1)
        return out
