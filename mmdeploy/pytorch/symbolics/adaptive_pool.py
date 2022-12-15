# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.core import SYMBOLIC_REWRITER


@SYMBOLIC_REWRITER.register_symbolic(
    'adaptive_avg_pool2d', is_pytorch=True, backend='ncnn')
def adaptive_avg_pool2d__ncnn(g, x, output_size):
    """Register ncnn symbolic function for `adaptive_avg_pool2d`.

    Align symbolic of adaptive_avg_pool2d in ncnn.
    """
    return g.op('mmdeploy::AdaptiveAvgPool2d', x, output_size)
