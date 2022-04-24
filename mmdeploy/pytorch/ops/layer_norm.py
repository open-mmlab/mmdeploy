# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


class LayerNormop(torch.autograd.Function):
    """Create onnx::LayerNorm op."""

    @staticmethod
    def forward(ctx, inp: Tensor, w: Tensor, b: Tensor, eps: float) -> Tensor:

        return torch.rand_like(inp)

    @staticmethod
    def symbolic(g, inp: Tensor, w: Tensor, b: Tensor, eps: float):
        w.setDebugName('layernorm_weight')
        b.setDebugName('layernorm_bias')

        return g.op(
            'mmdeploy::LayerNorm', inp, w, b, affine_i=1, epsilon_f=eps)


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.LayerNorm.forward', backend=Backend.NCNN.value)
def layernorm__forward__ncnn(ctx, self, inp):
    """Rewrite `forward` of LayerNorm used in vision_transformer for ncnn
    backend.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (LayerNorm): The instance of the class LayerNorm.
        inp (Tensor): Input features of shape (N, Cin, H, W).
    Returns:
        out (Tensor): A feature map output from MultiHeadAttention. The tensor
        shape (N, Cout, H, W).
    """

    return LayerNormop.apply(inp, self.weight.data, self.bias.data, self.eps)
