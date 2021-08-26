import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.Tensor.__getattribute__', backend='ncnn')
def getattribute_static(ctx, self, name):
    ret = ctx.origin_func(self, name)
    if name == 'shape':
        ret = torch.Size([int(s) for s in ret])
    return ret
