# Copyright (c) OpenMMLab. All rights reserved.
import sys

from .backend_manager import IPUManager


def is_available():
    """Check whether popart is installed & IPU device is available.

    Returns:
        bool: True if popart is installed & IPU device is available.
    """

    try:
        if 'onnx' in sys.modules.keys():
            del sys.modules['onnx']
            # del onnx
            import onnx
            import popart
        else:
            import popart

        deviceManager = popart.DeviceManager()
        device = deviceManager.acquireAvailableDevice(1)
        return True
    except Exception as e:
        print('IPU environment is not set', str(e))
        return False


__all__ = ['IPUManager']

if is_available():
    try:
        from .converter import onnx_to_popef
        from .wrapper import IPUWrapper
        __all__ += ['IPUWrapper', 'onnx_to_popef']
    except Exception as e:
        print('ipu import error ', e)
        pass
