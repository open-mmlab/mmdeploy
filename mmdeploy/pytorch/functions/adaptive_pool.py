# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend, get_root_logger, is_dynamic_shape


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.adaptive_avg_pool2d')
def adaptive_avg_pool2d__default(input, output_size):
    """Rewrite `adaptive_avg_pool2d` for default backend."""
    ctx = FUNCTION_REWRITER.get_context()
    output_size = _pair(output_size)
    if int(output_size[0]) == int(output_size[1]) == 1:
        out = ctx.origin_func(input, output_size)
    else:
        deploy_cfg = ctx.cfg
        is_dynamic_flag = is_dynamic_shape(deploy_cfg)
        if is_dynamic_flag:
            logger = get_root_logger()
            logger.warning('`adaptive_avg_pool2d` would be '
                           'replaced to `avg_pool2d` explicitly')
        size = input.shape[2:]
        k = [int(size[i] / output_size[i]) for i in range(0, len(size))]
        out = F.avg_pool2d(
            input,
            kernel_size=k,
            stride=k,
            padding=0,
            ceil_mode=False,
            count_include_pad=False)
    return out


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.adaptive_avg_pool2d',
    backend=Backend.NCNN.value)
@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.adaptive_avg_pool2d',
    backend=Backend.TORCHSCRIPT.value)
def adaptive_avg_pool2d__ncnn(input, output_size):
    ctx = FUNCTION_REWRITER.get_context()
    """Rewrite `adaptive_avg_pool2d` for ncnn and torchscript backend."""
    return ctx.origin_func(input, output_size)
