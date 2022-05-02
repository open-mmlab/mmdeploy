# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from subprocess import call
from typing import Union

import onnx

from .init_plugins import get_onnx2ncnn_path


def from_onnx(onnx_model: Union[onnx.ModelProto, str],
              output_file_prefix: str):
    """Convert ONNX to ncnn.

    The inputs of ncnn include a model file and a weight file. We need to use
    a executable program to convert the `.onnx` file to a `.param` file and
    a `.bin` file. The output files will save to work_dir.

    Example:
        >>> from mmdeploy.apis.ncnn import from_onnx
        >>> onnx_path = 'work_dir/end2end.onnx'
        >>> save_param = 'work_dir/end2end.param'
        >>> save_bin = 'work_dir/end2end.bin'
        >>> from_onnx(onnx_path, save_param, save_bin)

    Args:
        onnx_path (ModelProto|str): The path of the onnx model.
        output_file_prefix (str): The path to save the output ncnn file.
    """

    if not isinstance(onnx_model, str):
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)
    else:
        onnx_path = onnx_model

    save_param = output_file_prefix + '.param'
    save_bin = output_file_prefix + '.bin'

    onnx2ncnn_path = get_onnx2ncnn_path()
    call([onnx2ncnn_path, onnx_path, save_param, save_bin])
