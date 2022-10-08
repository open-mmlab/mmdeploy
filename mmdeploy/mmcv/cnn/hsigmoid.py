# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


class hsigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        return torch.nn.functional.hardsigmoid(x)

    @staticmethod
    def symbolic(g, x):
        return g.op('mmdeploy::HardSigmoid', x, alpha_f=0.5, beta_f=0.5)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.cnn.bricks.hsigmoid.HSigmoid.forward',
    backend=Backend.NCNN.value)
def hsigmoid__forward__ncnn(ctx, self, x):
    return hsigmoid.apply(x)
