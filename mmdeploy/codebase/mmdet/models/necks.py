import torch

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.necks.ssd_neck.L2Norm.forward')
def l2norm__forward__default(ctx, self, x):
    return torch.nn.functional.normalize(
        x, dim=1) * self.weight[None, :, None, None]
