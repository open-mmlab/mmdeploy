# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa


def is_available():
    """Torchscript available.

    Returns:
        bool: Always True.
    """
    return True


if is_available():
    from .wrapper import TorchscriptWrapper

    __all__ = ['TorchscriptWrapper']
