# Copyright (c) OpenMMLab. All rights reserved.
from torch.onnx import symbolic_helper

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend


@symbolic_helper.parse_args('v')
def gelu__ncnn_pt111(g, self):
    """gelu for torch<=1.12."""
    return g.op('mmdeploy::Gelu', self)


@SYMBOLIC_REWRITER.register_symbolic(
    'gelu', is_pytorch=True, backend=Backend.NCNN.value)
def gelu__ncnn(g, self, approximate: str = 'none'):
    """Support export GELU with ncnn backend."""
    return gelu__ncnn_pt111(g, self)
