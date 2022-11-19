# Copyright (c) OpenMMLab. All rights reserved.


def is_available():
    """Check whether popart is installed & IPU device is available.

    Returns:
        bool: True if popart is installed & IPU device is available.
    """

    try:
        import popart
        device = popart.DeviceManager.acquireAvailableDevice()
        import popconverter

        return True
    except Exception as e:
        print('IPU environment is not set')
        print(e)
        return False


__all__ = []

if is_available():
    try:
        from .wrapper import IPUWrapper
        from .converter import onnx_to_popef

        __all__ += ['IPUWrapper', 'onnx_to_popef']
    except Exception as e:
        print(e)
        pass
