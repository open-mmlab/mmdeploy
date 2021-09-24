import importlib
import os.path as osp

from .init_plugins import get_ops_path


def is_available():
    """Check whether onnxruntime and its custom ops are installed.

    Returns:
        bool: True if onnxruntime package is installed and its
        custom ops are compiled.
    """
    onnxruntime_op_path = get_ops_path()
    if not osp.exists(onnxruntime_op_path):
        return False
    return importlib.util.find_spec('onnxruntime') is not None


if is_available():
    from .onnxruntime_utils import ORTWrapper
    __all__ = ['get_ops_path', 'ORTWrapper']
