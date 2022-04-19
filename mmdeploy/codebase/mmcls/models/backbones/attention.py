# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from torch.nn import Linear
from mmcls.models.utils import channel_shuffle

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend

class MultiHeadAttentionop(torch.autograd.Function):
    """Create onnx::MultiHeadAttention op.
    """

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, 
                q_weight: Tensor, q_bias: Tensor, 
                k_weight: Tensor, k_bias: Tensor, 
                v_weight: Tensor, v_bias: Tensor,
                o_weight: Tensor, o_bias: Tensor,
                embed_dims: int, num_heads: int) -> Tensor:

        # head_dims = embed_dims // num_heads
        # scale = head_dims**-0.5

        # print("forward")
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)

        # q = q @ q_weight + q_bias
        # # q [1, 145, 768] to [1, 12, 145, 64]
        # q = q.reshape(1, -1, num_heads, head_dims).permute(0, 2, 1, 3)

        # k = k @ k_weight + k_bias
        # # k reshape to [1, 145, 12, 64], permute to [1, 12, 64, 145]
        # k.reshape(1, -1, num_heads, head_dims).permute(0, 2, 3, 1)
        
        # v = v @ v_weight + v_bias
        # # v reshape and permute to [1, 12, 145, 64]
        # v = v.reshape(1, -1, num_heads, head_dims).permute(0, 2, 1, 3)

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        
        # # attn shape = [1, 12, 145, 145]
        # attn = q @ k * scale
        # attn = attn.softmax(dim=-1)

        # # x shape = [1, 12, 145, 64]
        # x = attn @ v
        # # x shape = [1, 145, 768]
        # x = x.permute(0, 2, 1, 3).reshape(1, -1, embed_dims)
        # x = x @ o_weight + o_bias

        return torch.rand_like(q)


    @staticmethod
    def symbolic(g, q: Tensor, k: Tensor, v: Tensor, 
                q_weight: Tensor, q_bias: Tensor, 
                k_weight: Tensor, k_bias: Tensor, 
                v_weight: Tensor, v_bias: Tensor,
                o_weight: Tensor, o_bias: Tensor,
                embed_dims: int, num_heads: int):

        q_weight.setDebugName("q_weight")
        q_bias.setDebugName("q_bias")

        k_weight.setDebugName("k_weight")
        k_bias.setDebugName("k_bias")

        v_weight.setDebugName("v_weight")
        v_bias.setDebugName("v_bias")

        o_weight.setDebugName("o_weight")
        o_bias.setDebugName("o_bias")
        
        return g.op(
            'mmdeploy::MultiHeadAttention',
            q, k, v,
            q_weight, q_bias,
            k_weight, k_bias,
            v_weight, v_bias,
            o_weight, o_bias,
            embed_dim_i = embed_dims,
            num_head_i = num_heads)


# ncnn have implemented MultiheadAttention, onnx would split this opr. So the model needs to rewrite.
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.utils.attention.MultiheadAttention.forward',
    backend=Backend.NCNN.value)
def multiheadattention__forward__ncnn(ctx, self, qkv_input):
    """Rewrite `forward` of MultiheadAttention used in vision_transformer for ncnn
    backend.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (InvertedResidual): The instance of the class InvertedResidual.
        x (Tensor): Input features of shape (N, Cin, H, W).
    Returns:
        out (Tensor): A feature map output from MultiHeadAttention. The tensor
        shape (N, Cout, H, W).
    """

    # split qkv weight and bias
    # [768, 2304] => 3 * [768, 768]
    qkv_weight = self.qkv.weight.data.reshape(self.input_dims, 3, self.embed_dims).permute(1, 0, 2)
    q_weight = qkv_weight[0]
    k_weight = qkv_weight[1]
    v_weight = qkv_weight[2]

    # [2304] => 3 * [768]
    qkv_bias = self.qkv.bias.data.reshape(3, self.embed_dims)
    q_bias = qkv_bias[0]
    k_bias = qkv_bias[1]
    v_bias = qkv_bias[2]

    # out weight and bias
    o_weight = self.proj.weight.data
    o_bias = self.proj.bias.data

    out = MultiHeadAttentionop.apply(qkv_input, qkv_input, qkv_input, 
        q_weight, q_bias, 
        k_weight, k_bias, 
        v_weight, v_bias, 
        o_weight, o_bias, 
        self.embed_dims, self.num_heads)
    return out
