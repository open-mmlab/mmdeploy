# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


class hsigmoid(torch.autograd.Function):
    """Rewrite this op because the param 'lower' and 'upper' in ncnn are fixed
    while 'min_value' and 'max_value' are configurable in mmcv."""

    @staticmethod
    def forward(ctx, x, bias: float, divisor: float, min_value, max_value):

        return torch.nn.functional.hardsigmoid(x)

    @staticmethod
    def symbolic(g, x, bias: float, divisor: float, min_value, max_value):

        temp = 1.0 / divisor
        x = g.op('Add', x, g.op('Constant', value_t=torch.tensor([bias])))
        x = g.op('Mul', x, g.op('Constant', value_t=torch.tensor([temp])))
        return g.op('Clip', x,
                    g.op('Constant', value_t=torch.Tensor([min_value])),
                    g.op('Constant', value_t=torch.Tensor([max_value])))


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcv.cnn.bricks.hsigmoid.HSigmoid.forward',
    backend=Backend.NCNN.value)
def hsigmoid__forward__ncnn(ctx, self, x):
    return hsigmoid.apply(x, self.bias, self.divisor, self.min_value,
                          self.max_value)
