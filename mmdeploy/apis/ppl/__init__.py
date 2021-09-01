import importlib


def is_available():
    """Check whether ppl is installed."""
    return importlib.util.find_spec('pyppl') is not None


if is_available():
    from .ppl_utils import PPLWrapper, register_engines
    __all__ = ['register_engines', 'PPLWrapper']
