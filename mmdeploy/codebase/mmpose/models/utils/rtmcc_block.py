# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn.functional as F
from mmpose.models.utils import rope

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.utils.rtmcc_block.ScaleNorm.forward', backend='ncnn')
def scalenorm__forward__ncnn(self, x):
    """Rewrite `scalenorm` for ncnn backend.

    Rewrite scalenorm to avoid FP16 exceed in ncnn Android platform.
    """
    # The one-dim of Fubinious norm is equal to L2Norm.
    # Set p=2 explicitly to map torch.norm to ReduceL2 onnx op,
    # which will avoid FP16 exceed.
    norm = torch.norm(x, dim=2, keepdim=True)
    norm = norm * self.scale
    # Rewrite for ncnn binaryop broadcast.
    norm = norm.clamp(min=self.eps)
    return (x.unsqueeze(2) / norm.unsqueeze(2)).squeeze(2) * self.g


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.utils.rtmcc_block.RTMCCBlock._forward', backend='ncnn')
def rtmccblock___forward_ncnn(self, inputs):
    """Rewrite `_forward` of RTMBlock for ncnn backend.

    Rewrite the matmul and avoid unbind for ncnn backend.
    """
    if self.attn_type == 'self-attn':
        x = inputs
    else:
        x, k, v = inputs

    x = self.ln(x)
    uv = self.uv(x)
    if self.attn_type == 'self-attn':
        uv = self.act_fn(uv)
        u = uv[..., :self.e]
        v = uv[..., self.e:2 * self.e]
        base = uv[..., 2 * self.e:2 * self.e + self.s]

        q = (base.unsqueeze(1) * self.gamma[None, None, 0:1, :] +
             self.beta[None, None, 0:1, :]).squeeze(1)
        k = (base.unsqueeze(1) * self.gamma[None, None, 1:2, :] +
             self.beta[None, None, 1:2, :]).squeeze(1)

        if self.pos_enc:
            q = rope(q, dim=1)
            k = rope(k, dim=1)
    else:
        u, q = torch.split(self.act_fn(uv), [self.e, self.s], dim=-1)

        k = self.k_fc(k)
        v = self.v_fc(v)

        if self.pos_enc:
            q = rope(q, 1)
            k = rope(k, 1)
    qk = torch.bmm(q, k.permute(0, 2, 1))
    if self.use_rel_bias:
        if self.attn_type == 'self-attn':
            bias = self.rel_pos_bias(q.size(1))
        else:
            bias = self.rel_pos_bias(q.size(1), k.size(1))
        qk += bias[:, :q.size(1), :k.size(1)]

    kernel = torch.square(F.relu(qk / self.sqrt_s))
    if self.dropout_rate > 0.:
        kernel = self.dropout(kernel)

    x = u * torch.bmm(kernel, v)
    x = self.o(x)

    return x


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.utils.rtmcc_block.Scale.forward', backend='ncnn')
def scale__forward_ncnn(self, x):
    """Rewrite `forward` of Scale for ncnn backend.

    Adapt the shape to avoid ncnn BinaryOp seg fault.
    """
    x = x.unsqueeze(1)
    scale = self.scale[None, None, None, :]
    return (x * scale).squeeze(1)
