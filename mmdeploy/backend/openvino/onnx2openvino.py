# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import subprocess
from subprocess import PIPE, CalledProcessError, run
from typing import Dict, List, Optional, Union

import mmcv
import torch

from mmdeploy.utils import get_root_logger
from .utils import ModelOptimizerOptions


def get_mo_command() -> str:
    """Checks for possible commands to run Model Optimizer. The following
    commands will be tested:

        'mo.py' - if you installed OpenVINO using the installer.
        'mo' - if you installed OpenVINO with pip.

    Returns:
        str: Command to run Model Optimizer. If it is not available,
            the empty string "" will be returned.
    """
    mo_command = ''
    mo_commands = ['mo.py', 'mo']
    for command in mo_commands:
        is_available = True
        try:
            run(f'{command} -h',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True,
                check=True)
        except CalledProcessError:
            is_available = False
        if is_available:
            mo_command = command
    return mo_command


def get_output_model_file(onnx_path: str, work_dir: str) -> str:
    """Returns the path to the .xml file with export result.

    Args:
        onnx_path (str): The path to the onnx model.
        work_dir (str): The path to the directory for saving the results.

    Returns:
        str: The path to the file where the export result will be located.
    """
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    file_name = osp.splitext(osp.split(onnx_path)[1])[0]
    model_xml = osp.join(work_dir, file_name + '.xml')
    return model_xml


def onnx2openvino(input_info: Dict[str, Union[List[int], torch.Size]],
                  output_names: List[str],
                  onnx_path: str,
                  work_dir: str,
                  mo_options: Optional[ModelOptimizerOptions] = None):
    """Convert ONNX to OpenVINO.

    Examples:
        >>> from mmdeploy.backend.openvino.onnx2openvino import onnx2openvino
        >>> input_info = {'input': [1,3,800,1344]}
        >>> output_names = ['dets', 'labels']
        >>> onnx_path = 'work_dir/end2end.onnx'
        >>> work_dir = 'work_dir'
        >>> onnx2openvino(input_info, output_names, onnx_path, work_dir)

    Args:
        input_info (Dict[str, Union[List[int], torch.Size]]):
            The shape of each input.
        output_names (List[str]): Output names. Example: ['dets', 'labels'].
        onnx_path (str): The path to the onnx model.
        work_dir (str): The path to the directory for saving the results.
        mo_options (None | ModelOptimizerOptions): The class with
            additional arguments for the Model Optimizer.
    """
    input_names = ','.join(input_info.keys())
    input_shapes = ','.join(str(list(elem)) for elem in input_info.values())
    output = ','.join(output_names)

    mo_command = get_mo_command()
    is_mo_available = bool(mo_command)
    if not is_mo_available:
        raise RuntimeError(
            'OpenVINO Model Optimizer is not found or configured improperly')

    mo_args = f'--input_model="{onnx_path}" '\
              f'--output_dir="{work_dir}" ' \
              f'--output="{output}" ' \
              f'--input="{input_names}" ' \
              f'--input_shape="{input_shapes}" '
    if mo_options is not None:
        mo_args += mo_options.get_options()

    command = f'{mo_command} {mo_args}'

    logger = get_root_logger()
    logger.info(f'Args for Model Optimizer: {command}')
    mo_output = run(command, stdout=PIPE, stderr=PIPE, shell=True, check=True)
    logger.info(mo_output.stdout.decode())
    logger.debug(mo_output.stderr.decode())

    model_xml = get_output_model_file(onnx_path, work_dir)
    logger.info(f'Successfully exported OpenVINO model: {model_xml}')
