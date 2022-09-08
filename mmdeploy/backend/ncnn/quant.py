# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from subprocess import call
from typing import List

import mmengine

from .init_plugins import get_ncnn2int8_path


def get_quant_model_file(onnx_path: str, work_dir: str) -> List[str]:
    """Returns the path to quant onnx and table with export result.

    Args:
        onnx_path (str): The path to the fp32 onnx model.
        work_dir (str): The path to the directory for saving the results.

    Returns:
        List[str]: The path to the files where the export result will be
            located.
    """
    mmengine.mkdir_or_exist(osp.abspath(work_dir))
    base_name = osp.splitext(osp.split(onnx_path)[1])[0]
    quant_onnx = osp.join(work_dir, base_name + '_quant.onnx')
    quant_table = osp.join(work_dir, base_name + '.table')
    quant_param = osp.join(work_dir, base_name + '_int8.param')
    quant_bin = osp.join(work_dir, base_name + '_int8.bin')
    return [quant_onnx, quant_table, quant_param, quant_bin]


def ncnn2int8(param: str, bin: str, table: str, int8_param: str,
              int8_bin: str):
    """Convert ncnn float model to quantized model.

    The inputs of ncnn include float model and weight file. We need to use
    a executable program to convert the float model to int8 model with
    calibration table.

    Example:
        >>> from mmdeploy.backend.ncnn.quant import ncnn2int8
        >>> param = 'work_dir/end2end.param'
        >>> bin = 'work_dir/end2end.bin'
        >>> table = 'work_dir/end2end.table'
        >>> int8_param = 'work_dir/end2end_int8.param'
        >>> int8_bin = 'work_dir/end2end_int8.bin'
        >>> ncnn2int8(param, bin, table, int8_param, int8_bin)

    Args:
        param (str): The path of ncnn float model graph.
        bin (str): The path of ncnn float weight model weight.
        table (str): The path of ncnn calibration table.
        int8_param (str):  The path of ncnn low bit model graph.
        int8_bin (str): The path of ncnn low bit weight model weight.
    """

    ncnn2int8 = get_ncnn2int8_path()

    call([ncnn2int8, param, bin, int8_param, int8_bin, table])
