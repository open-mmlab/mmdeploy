# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import Backend


@SYMBOLIC_REWRITER.register_symbolic(
    'gelu', is_pytorch=True, arg_descriptors=['v'], backend=Backend.NCNN.value)
def gelu__ncnn(ctx, g, self):
    """Support export GELU with ncnn backend."""
    return g.op('mmdeploy::Gelu', self)
