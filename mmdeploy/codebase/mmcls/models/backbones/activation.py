# Copyright (c) OpenMMLab. All rights reserved.
import torch
import mmcv
from torch import Tensor
from torch.nn import Linear

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend

class GELUop(torch.autograd.Function):
    """Create onnx::GELU op.
    """

    @staticmethod
    def forward(ctx, inp: Tensor) -> Tensor:
        return torch.rand_like(inp)


    @staticmethod
    def symbolic(g, inp: Tensor):
        return g.op('mmdeploy::Gelu', inp)


# ncnn have implemented MultiheadAttention, onnx would split this opr. So the model needs to rewrite.
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.GELU.forward',
    backend=Backend.NCNN.value)
def gelu__forward__ncnn(ctx, self, inp):
    """Rewrite `forward` of MultiheadAttention used in vision_transformer for ncnn
    backend.

    Args:
        ctx (ContextCaller): The context with additional information.
        self (InvertedResidual): The instance of the class InvertedResidual.
        inp (Tensor): Input features of shape (N, Cin, H, W).
    Returns:
        out (Tensor): A feature map output from MultiHeadAttention. The tensor
        shape (N, Cout, H, W).
    """

    out = GELUop.apply(inp)
    return out
