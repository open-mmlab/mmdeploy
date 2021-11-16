import importlib


def is_available():
    """Check whether ppl is installed.

    Returns:
        bool: True if ppl package is installed.
    """
    return importlib.util.find_spec('pyppl') is not None


if is_available():
    from .ppl_utils import PPLWrapper, register_engines
    from .onnx2ppl import onnx2ppl
    __all__ = ['register_engines', 'PPLWrapper', 'onnx2ppl']
