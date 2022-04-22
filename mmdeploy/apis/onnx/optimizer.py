# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable

from torch.onnx import OperatorExportTypes

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter('torch.onnx.utils._model_to_graph')
def model_to_graph__custom_optimizer(
        ctx,
        model,
        args,
        verbose=False,
        input_names=None,
        output_names=None,
        operator_export_type=OperatorExportTypes.ONNX,
        example_outputs=None,
        _retain_param_name=False,
        do_constant_folding=True,
        _disable_torch_constant_prop=False,
        fixed_batch_size=False,
        training=None,
        use_new_jit_passes=True,
        dynamic_axes=None):
    """Rewriter of _model_to_graph, add custom passes."""
    graph, params_dict, torch_out = ctx.origin_func(
        model,
        args,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        operator_export_type=operator_export_type,
        example_outputs=example_outputs,
        _retain_param_name=_retain_param_name,
        do_constant_folding=do_constant_folding,
        _disable_torch_constant_prop=_disable_torch_constant_prop,
        fixed_batch_size=fixed_batch_size,
        training=training,
        use_new_jit_passes=use_new_jit_passes,
        dynamic_axes=dynamic_axes)

    custom_passes = getattr(ctx, 'onnx_custom_passes', None)

    if custom_passes is not None:
        assert isinstance(
            custom_passes, Callable
        ), f'Expect a callable onnx_custom_passes, get {type(custom_passes)}.'
        graph, params_dict, torch_out = custom_passes(graph, params_dict,
                                                      torch_out)

    return graph, params_dict, torch_out
