# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils.constants import Backend


class ScaledDotProductAttentionTRT(torch.autograd.Function):
    """Caller of scale dot product attention."""

    @staticmethod
    def forward(ctx,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                attn_mask: Optional[Tensor] = None):
        """forward function."""
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += attn_mask

        attn = attn.softmax(-1)

        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

    @staticmethod
    def symbolic(g, q, k, v, mask):
        """Symbolic function."""
        inputs = [q, k, v]
        if mask is not None:
            inputs += [mask]
        return g.op(
            'mmdeploy::ScaledDotProductAttentionTRT', *inputs, outputs=2)


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional._scaled_dot_product_attention',
    backend=Backend.TENSORRT.value)
def _scaled_dot_product_attention__tensorrt(q: Tensor,
                                            k: Tensor,
                                            v: Tensor,
                                            attn_mask: Optional[Tensor] = None,
                                            dropout_p: float = 0.0,
                                            **kwargs) -> Tuple[Tensor, Tensor]:
    """Rewrite for custom ops."""
    return ScaledDotProductAttentionTRT.apply(q, k, v, attn_mask)


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.scaled_dot_product_attention',
    backend=Backend.DEFAULT.value)
def scaled_dot_product_attention__default(query,
                                          key,
                                          value,
                                          attn_mask=None,
                                          dropout_p=0.,
                                          scale=None,
                                          is_causal=False):
    """Rewrite to export to onnx on torch>=2.0.0."""
    scale = scale or query.size(-1)**0.5
    if is_causal and attn_mask is not None:
        attn_mask = torch.ones(
            query.size(-2), key.size(-2), dtype=torch.bool).tril(diagonal=0)
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf'))

    attn_weight = query @ key.transpose(-2, -1) / scale
    if attn_mask is not None:
        attn_weight += attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value
