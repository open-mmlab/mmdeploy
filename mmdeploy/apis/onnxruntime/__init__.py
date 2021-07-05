from .init_plugins import get_ops_path

__all__ = ['get_ops_path']


def is_available():
    import os.path as osp
    tensorrt_op_path = get_ops_path()
    if not osp.exists(tensorrt_op_path):
        return False

    import importlib
    return importlib.util.find_spec('onnxruntime') is not None
