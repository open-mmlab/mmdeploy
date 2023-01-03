# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp


def get_ops_path() -> str:
    """Get path of the torchscript extension library.

    Returns:
        str: A path of the torchscript extension library.
    """
    from mmdeploy.utils import get_file_path
    candidates = [
        '../../lib/libmmdeploy_torchscript_ops.so',
        '../../lib/mmdeploy_torchscript_ops.dll',
        '../../../build/lib/libmmdeploy_torchscript_ops.so',
        '../../../build/bin/*/mmdeploy_torchscript_ops.dll'
    ]
    return get_file_path(osp.dirname(__file__), candidates)


def ops_available() -> bool:
    """Return whether ops are available.

    Returns:
        bool: Whether ops are available.
    """
    return osp.exists(get_ops_path())
