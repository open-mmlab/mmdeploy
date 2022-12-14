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
    func_name='mmcv.cnn.bricks.transformer.MultiheadAttention.forward',
    backend=Backend.NCNN.value)
def multiheadattention__forward__ncnn(self,
                                      query,
                                      key=None,
                                      value=None,
                                      identity=None,
                                      query_pos=None,
                                      key_pos=None,
                                      attn_mask=None,
                                      key_padding_mask=None,
                                      **kwargs):
    """Rewrite `forward` of MultiheadAttention used in vision_transformer for
    ncnn backend.

    Args:
        query (Tensor): The input query with shape [num_queries, bs,
            embed_dims] if self.batch_first is False, else
            [bs, num_queries embed_dims].
        key (Tensor): The key tensor with shape [num_keys, bs,
            embed_dims] if self.batch_first is False, else
            [bs, num_keys, embed_dims] .
            If None, the ``query`` will be used. Defaults to None.
        value (Tensor): The value tensor with same shape as `key`.
            Same in `nn.MultiheadAttention.forward`. Defaults to None.
            If None, the `key` will be used.
        identity (Tensor): This tensor, with the same shape as x,
            will be used for the identity link.
            If None, `x` will be used. Defaults to None.
        query_pos (Tensor): The positional encoding for query, with
            the same shape as `x`. If not None, it will
            be added to `x` before forward function. Defaults to None.
        key_pos (Tensor): The positional encoding for `key`, with the
            same shape as `key`. Defaults to None. If not None, it will
            be added to `key` before forward function. If None, and
            `query_pos` has the same shape as `key`, then `query_pos`
            will be used for `key_pos`. Defaults to None.
        attn_mask (Tensor): ByteTensor mask with shape [num_queries,
            num_keys]. Same in `nn.MultiheadAttention.forward`.
            Defaults to None.
        key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
            Defaults to None.
    Returns:
        Tensor: forwarded results with shape
        [bs, num_queries embed_dims].
    """
    if key is None:
        key = query
    if value is None:
        value = key
    if identity is None:
        identity = query
    if key_pos is None:
        if query_pos is not None:
            # use query_pos if key_pos is not available
            if query_pos.shape == key.shape:
                key_pos = query_pos
    if query_pos is not None:
        query = query + query_pos
    if key_pos is not None:
        key = key + key_pos

    assert query is key and key is value, 'only support query==key==value'
    assert self.batch_first, 'only support batch on first dim'
    assert attn_mask is None
    assert key_padding_mask is None

    # split qkv weight and bias
    qkv_weight = self.attn.in_proj_weight.data.reshape(3, -1, self.embed_dims)

    q_weight = qkv_weight[0]
    k_weight = qkv_weight[1]
    v_weight = qkv_weight[2]

    qkv_bias = self.attn.in_proj_bias.data.reshape(3, self.embed_dims)
    q_bias = qkv_bias[0]
    k_bias = qkv_bias[1]
    v_bias = qkv_bias[2]

    # out weight and bias
    o_weight = self.attn.out_proj.weight.data
    o_bias = self.attn.out_proj.bias.data
    # export to MultiHeadAttention in ncnn
    out = MultiHeadAttentionop.apply(query, key, value, q_weight, q_bias,
                                     k_weight, k_bias, v_weight, v_bias,
                                     o_weight, o_bias, self.embed_dims,
                                     self.num_heads)
    return identity + self.dropout_layer(self.proj_drop(out))
