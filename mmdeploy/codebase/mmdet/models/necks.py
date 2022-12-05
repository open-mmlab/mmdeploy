# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend, get_root_logger


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.necks.ssd_neck.L2Norm.forward')
def l2norm__forward__default(self, x):
    """Default rewriter for l2norm.

    Implement with functinoal.normalize .
    """
    return torch.nn.functional.normalize(
        x, dim=1) * self.weight[None, :, None, None]


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdet.models.necks.ssd_neck.L2Norm.forward',
    backend=Backend.TENSORRT.value)
def l2norm__forward__tensorrt(self, x):
    """rewrite `l2norm` for TensorRT.

    TensorRT7 does not support dynamic clamp, which is used in normalize.
    """
    ctx = FUNCTION_REWRITER.get_context()
    logger = get_root_logger()
    trt_version_major = 8
    try:
        import tensorrt as trt
        from packaging import version
        trt_version = version.parse(trt.__version__)
        trt_version_major = trt_version.major
    except Exception:
        logger.warning('Can not get TensorRT version.')
    if trt_version_major >= 8:
        return l2norm__forward__default(self, x)
    else:
        return ctx.origin_func(self, x)
