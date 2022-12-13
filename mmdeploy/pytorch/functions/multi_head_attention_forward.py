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
def _scaled_dot_product_attention__tensorrt(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Rewrite for custom ops."""
    return ScaledDotProductAttentionTRT.apply(q, k, v, attn_mask)
