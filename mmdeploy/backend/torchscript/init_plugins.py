import glob
import os.path as osp
import platform


def get_ops_path() -> str:
    """Get path of the torchscript extension library.

    Returns:
        str: A path of the torchscript extension library.
    """
    wildcard = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            '../../../build/lib/libmmdeploy_torchscript_ops.so'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path


def get_optimizer_path() -> str:
    """Get ts_optimizer path.

    Returns:
        str: A path of ts_optimizer tool.
    """
    wildcard = osp.abspath(
        osp.join(osp.dirname(__file__), '../../../build/bin/ts_optimizer'))
    if platform.system() == 'Windows':
        wildcard += '.exe'

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path
