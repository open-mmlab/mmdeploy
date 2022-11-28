# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.utils import get_root_logger


def optimize_onnx(ctx, graph, params_dict, torch_out):
    """The optimize callback of the onnx model."""
    logger = get_root_logger()
    logger.info('Execute onnx optimize passes.')
    try:
        from mmdeploy.backend.torchscript import ts_optimizer
        ts_optimizer.onnx._jit_pass_merge_shape_concate(graph)
        ts_optimizer.onnx._jit_pass_onnx_peephole(graph)
        ts_optimizer.onnx._jit_pass_flatten_cls_head(graph)
        ts_optimizer.onnx._jit_pass_fuse_select_assign(graph, params_dict)
        ts_optimizer.onnx._jit_pass_common_subgraph_elimination(
            graph, params_dict)
    except ImportError:
        logger.warning(
            'Can not optimize model, please build torchscipt extension.\n'
            'More details: '
            'https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/experimental/onnx_optimizer.md'  # noqa
        )
    return graph, params_dict, torch_out
