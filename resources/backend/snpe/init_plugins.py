# Copyright (c) OpenMMLab. All rights reserved.
import shutil


def get_onnx2dlc_path() -> str:
    """Get snpe-onnx-to-dlc path.

    Returns:
        str: A path of snpe-onnx-to-dlc tool.
    """
    return shutil.which('snpe-onnx-to-dlc')
