import importlib
import os.path as osp

from .init_plugins import get_onnx2ncnn_path, get_ops_path

__all__ = ['get_ops_path', 'get_onnx2ncnn_path']


def is_available():
    """Check whether ncnn with extension is installed.

    Returns:
        bool: True if ncnn and its extension are installed.
    """
    ncnn_ops_path = get_ops_path()
    if not osp.exists(ncnn_ops_path):
        return False
    has_pyncnn = importlib.util.find_spec('ncnn') is not None
    has_pyncnn_ext = importlib.util.find_spec(
        'mmdeploy.apis.ncnn.ncnn_ext') is not None

    return has_pyncnn and has_pyncnn_ext


if is_available():
    from .ncnn_utils import NCNNWrapper

    __all__ += ['NCNNWrapper']
