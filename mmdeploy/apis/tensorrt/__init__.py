# flake8: noqa
from .init_plugins import get_ops_path, load_tensorrt_plugin


def is_available():
    import os.path as osp
    tensorrt_op_path = get_ops_path()
    if not osp.exists(tensorrt_op_path):
        return False

    import importlib
    return importlib.util.find_spec('tensorrt') is not None


if is_available():
    from .onnx2tensorrt import onnx2tensorrt
    from .tensorrt_utils import (TRTWrapper, load_trt_engine, onnx2trt,
                                 save_trt_engine)

    # load tensorrt plugin lib
    load_tensorrt_plugin()

    __all__ = [
        'onnx2trt', 'save_trt_engine', 'load_trt_engine', 'TRTWraper',
        'TRTWrapper', 'is_tensorrt_plugin_loaded', 'preprocess_onnx',
        'onnx2tensorrt'
    ]
