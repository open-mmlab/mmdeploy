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
            import popart
            import onnx
        else:
            import popart
        print('popart imported')
        deviceManager = popart.DeviceManager()
        device = deviceManager.acquireAvailableDevice(1)
        print('ipu device checked')
        return True
    except Exception as e:
        print('IPU environment is not set')
        print(e)
        return False


__all__ = []

if is_available():
    try:
        from .wrapper import IPUWrapper
        print('imported wrapper')
        from .converter import onnx_to_popef
        print('imported ipu wrapper & onnx_to_popef')
        __all__ += ['IPUWrapper', 'onnx_to_popef']
    except Exception as e:
        print('import error ', e)
        pass
