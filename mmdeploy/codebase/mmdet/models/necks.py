import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.necks.ssd_neck.L2Norm.forward')
def l2norm__forward__default(ctx, self, x):
    return torch.nn.functional.normalize(
        x, dim=1) * self.weight[None, :, None, None]


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.necks.ssd_neck.L2Norm.forward',
    backend=Backend.TENSORRT.value)
def l2norm__forward__tensorrt(ctx, self, x):
    """rewrite `l2norm` for TensorRT.

    TensorRT7 does not support dynamic clamp, which is used in normalize.
    """
    import tensorrt as trt
    from packaging import version
    trt_version = version.parse(trt.__version__)
    if trt_version.major >= 8:
        return l2norm__forward__default(ctx, self, x)
    else:
        return ctx.origin_func(self, x)
