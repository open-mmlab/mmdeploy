# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils.constants import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmocr.models.common.ScaledDotProductAttention.forward',
    backend=Backend.ONNXRUNTIME.value)
def scaled_dot_product_attention__forward(ctx, self, q, k, v, mask=None):
    """Rewrite `forward` of ScaledDotProductAttention for default backend.

    Replace `attn.masked_fill(mask == 0, float('-inf'))` with
    `attn.masked_fill(mask == 0, -1e34)`.
    """
    attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

    if mask is not None:
        # use float('-inf') may raise NAN for onnx
        attn = attn.masked_fill(mask == 0, -1e34)

    attn = self.dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)

    return output, attn
