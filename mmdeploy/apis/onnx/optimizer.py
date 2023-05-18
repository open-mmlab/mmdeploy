# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable

import torch

from mmdeploy.core import FUNCTION_REWRITER


def update_squeeze_unsqueeze_opset13_pass(graph, params_dict, torch_out):
    """Update Squeeze/Unsqueeze axes for opset13."""
    for node in graph.nodes():
        if node.kind() in ['onnx::Squeeze', 'onnx::Unsqueeze'] and \
                node.hasAttribute('axes'):
            axes = node['axes']
            axes_node = graph.create('onnx::Constant')
            axes_node.t_('value', torch.LongTensor(axes))
            node.removeAttribute('axes')
            node.addInput(axes_node.output())
            axes_node.insertBefore(node)
    return graph, params_dict, torch_out


@FUNCTION_REWRITER.register_rewriter('torch.onnx.utils._model_to_graph')
def model_to_graph__custom_optimizer(*args, **kwargs):
    """Rewriter of _model_to_graph, add custom passes."""
    ctx = FUNCTION_REWRITER.get_context()
    graph, params_dict, torch_out = ctx.origin_func(*args, **kwargs)
    if hasattr(ctx, 'opset'):
        opset_version = ctx.opset
    else:
        from mmdeploy.utils import get_ir_config
        opset_version = get_ir_config(ctx.cfg).get('opset_version', 11)
    if opset_version >= 13:
        graph, params_dict, torch_out = update_squeeze_unsqueeze_opset13_pass(
            graph, params_dict, torch_out)
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
