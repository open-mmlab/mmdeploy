import glob
import os


def get_ops_path():
    """Get NCNN custom ops library path."""
    wildcard = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../../build/lib/libmmlab_ncnn_ops.so'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path


def get_onnx2ncnn_path():
    """Get onnx2ncnn path."""
    wildcard = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '../../../build/bin/onnx2ncnn'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path
