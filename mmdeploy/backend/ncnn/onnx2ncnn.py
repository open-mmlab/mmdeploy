# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from subprocess import call
from typing import List

import mmcv

from .init_plugins import get_onnx2ncnn_path


def get_output_model_file(onnx_path: str, work_dir: str) -> List[str]:
    """Returns the path to the .param, .bin file with export result.

    Args:
        onnx_path (str): The path to the onnx model.
        work_dir (str): The path to the directory for saving the results.

    Returns:
        List[str]: The path to the files where the export result will be
            located.
    """
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    file_name = osp.splitext(osp.split(onnx_path)[1])[0]
    save_param = osp.join(work_dir, file_name + '.param')
    save_bin = osp.join(work_dir, file_name + '.bin')
    return [save_param, save_bin]


def onnx2ncnn(onnx_path: str, save_param: str, save_bin: str):
    """Convert ONNX to ncnn.

    The inputs of ncnn include a model file and a weight file. We need to use
    a executable program to convert the `.onnx` file to a `.param` file and
    a `.bin` file. The output files will save to work_dir.

    Args:
        onnx_path (str): The path of the onnx model.
        save_param (str): The path to save the output `.param` file.
        save_bin (str): The path to save the output `.bin` file.
    """

    onnx2ncnn_path = get_onnx2ncnn_path()

    call([onnx2ncnn_path, onnx_path, save_param, save_bin])
