# Copyright (c) OpenMMLab. All rights reserved.

import importlib


def is_available():
    """Check whether coremltools is installed.

    Returns:
        bool: True if coremltools package is installed.
    """
    return importlib.util.find_spec('coremltools') is not None


__all__ = []

if is_available():
    from . import ops
    from .torchscript2coreml import get_model_suffix
    from .wrapper import CoreMLWrapper
    __all__ += ['CoreMLWrapper', 'get_model_suffix', 'ops']
