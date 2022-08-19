# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils.constants import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional._scaled_dot_product_attention',
    backend=Backend.TENSORRT.value)
def _scaled_dot_product_attention__default(
    ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Rewrite `_scaled_dot_product_attention` to enable softmax."""
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    # add slice to enable softmax
    # TODO: Find the reason
    step = 500
    if attn.size(-1) > step:
        attn_max = attn[..., :step].max(-1, keepdim=True)[0]
        for i in range(step, attn.size(-1), step):
            attn_max_new = attn[..., i:i + step].max(-1, keepdim=True)[0]
            attn_max = attn_max.where(attn_max > attn_max_new, attn_max_new)
    else:
        attn_max = attn.max(-1, keepdim=True)[0]

    attn = attn - attn_max
    attn_exp = attn.exp()
    if attn_exp.size(-1) > step:
        attn_sum = attn_exp[..., :step].sum(-1, keepdim=True)
        for i in range(step, attn_exp.size(-1), step):
            attn_sum_new = attn_exp[..., i:i + step].sum(-1, keepdim=True)
            attn_sum += attn_sum_new
    else:
        attn_sum = attn_exp.sum(-1, keepdim=True)
    attn = attn_exp / attn_sum

    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn
