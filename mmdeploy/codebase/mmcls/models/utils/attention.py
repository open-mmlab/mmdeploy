# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


class MultiHeadAttentionop(torch.autograd.Function):
    """Create onnx::MultiHeadAttention op."""

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, q_weight: Tensor,
                q_bias: Tensor, k_weight: Tensor, k_bias: Tensor,
                v_weight: Tensor, v_bias: Tensor, o_weight: Tensor,
                o_bias: Tensor, embed_dims: int, num_heads: int) -> Tensor:
        return torch.rand_like(q)

    @staticmethod
    def symbolic(g, q: torch._C.Value, k: torch._C.Value, v: torch._C.Value,
                 q_weight: torch._C.Value, q_bias: torch._C.Value,
                 k_weight: torch._C.Value, k_bias: torch._C.Value,
                 v_weight: torch._C.Value, v_bias: torch._C.Value,
                 o_weight: torch._C.Value, o_bias: torch._C.Value,
                 embed_dims: int, num_heads: int):

        q_weight.setDebugName('q_weight')
        q_bias.setDebugName('q_bias')

        k_weight.setDebugName('k_weight')
        k_bias.setDebugName('k_bias')

        v_weight.setDebugName('v_weight')
        v_bias.setDebugName('v_bias')

        o_weight.setDebugName('o_weight')
        o_bias.setDebugName('o_bias')

        return g.op(
            'mmdeploy::MultiHeadAttention',
            q,
            k,
            v,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            embed_dim_i=embed_dims,
            num_heads_i=num_heads)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.utils.attention.MultiheadAttention.forward',
    backend=Backend.NCNN.value)
def multiheadattention__forward__ncnn(ctx, self, qkv_input):
    """Rewrite `forward` of MultiheadAttention used in vision_transformer for
    ncnn backend.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (MultiheadAttention): The instance of the class
        MultiheadAttention.
        x (Tensor): Input features of shape (N, Cin, H, W).
    Returns:
        out (Tensor): A feature map output from MultiHeadAttention. The tensor
        shape (N, Cout, H, W).
    """

    # split qkv weight and bias
    qkv_weight = self.qkv.weight.data.reshape(3, self.input_dims,
                                              self.embed_dims)

    q_weight = qkv_weight[0]
    k_weight = qkv_weight[1]
    v_weight = qkv_weight[2]

    qkv_bias = self.qkv.bias.data.reshape(3, self.embed_dims)
    q_bias = qkv_bias[0]
    k_bias = qkv_bias[1]
    v_bias = qkv_bias[2]

    # out weight and bias
    o_weight = self.proj.weight.data
    o_bias = self.proj.bias.data

    out = MultiHeadAttentionop.apply(qkv_input, qkv_input, qkv_input, q_weight,
                                     q_bias, k_weight, k_bias, v_weight,
                                     v_bias, o_weight, o_bias, self.embed_dims,
                                     self.num_heads)
    return out
