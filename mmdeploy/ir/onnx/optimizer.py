# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter('torch.onnx.utils._model_to_graph')
def model_to_graph__custom_optimizer(*args, **kwargs):
    """Rewriter of _model_to_graph, add custom passes."""
    ctx = FUNCTION_REWRITER.get_context()
    graph, params_dict, torch_out = ctx.origin_func(*args, **kwargs)

    custom_passes = getattr(ctx, 'onnx_custom_passes', None)

    if custom_passes is not None:
        assert isinstance(
            custom_passes, Callable
        ), f'Expect a callable onnx_custom_passes, get {type(custom_passes)}.'
        graph, params_dict, torch_out = custom_passes(ctx, graph, params_dict,
                                                      torch_out)

    return graph, params_dict, torch_out


@FUNCTION_REWRITER.register_rewriter(
    'torch._C._jit_pass_onnx_deduplicate_initializers', backend='tensorrt')
def jit_pass_onnx_deduplicate_initializers__disable(graph, param_dict, arg2):
    """This pass will disable TensorRT topk export.

    disable for TensorRT.
    """
    return param_dict


@FUNCTION_REWRITER.register_rewriter(
    'torch._C._jit_pass_onnx_autograd_function_process')
def jit_pass_onnx_autograd_function_process__disable(graph):
    """Disable process autograph function."""
    return
