# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.utils import get_root_logger


def optimize_onnx(graph, params_dict, torch_out):
    logger = get_root_logger()
    logger.info('Execute onnx optimize passes.')
    try:
        from mmdeploy.backend.torchscript import ts_optimizer
        ts_optimizer._jit_pass_merge_shape_concate(graph)
        ts_optimizer._jit_pass_onnx_peephole(graph)
    except Exception:
        pass

    return graph, params_dict, torch_out
