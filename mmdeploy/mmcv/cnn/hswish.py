# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


class hswish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        return torch.nn.functional.hardswish(x)

    @staticmethod
    def symbolic(g, x):
        return g.op('mmdeploy::HardSwish', x, alpha_f=0.1667, beta_f=0.5)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.cnn.bricks.hswish.HSwish.forward',
    backend=Backend.NCNN.value)
def hswish__forward__ncnn(ctx, self, x):
    return hswish.apply(x)
