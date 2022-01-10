# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import subprocess
from subprocess import CalledProcessError, run
from typing import Dict, List, Union

import mmcv
import torch

from mmdeploy.utils import get_root_logger


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
                  output_names: List[str], onnx_path: str, work_dir: str):
    """Convert ONNX to OpenVINO.

    Args:
        input_info (Dict[str, Union[List[int], torch.Size]]):
            The shape of each input.
        output_names (List[str]): Output names. Example: ['dets', 'labels'].
        onnx_path (str): The path to the onnx model.
        work_dir (str): The path to the directory for saving the results.
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
              f'--input_shape="{input_shapes}" ' \
              f'--disable_fusing '
    command = f'{mo_command} {mo_args}'

    logger = get_root_logger()
    logger.info(f'Args for Model Optimizer: {command}')
    mo_output = run(command, capture_output=True, shell=True, check=True)
    logger.info(mo_output.stdout.decode())
    logger.debug(mo_output.stderr.decode())

    model_xml = get_output_model_file(onnx_path, work_dir)
    logger.info(f'Successfully exported OpenVINO model: {model_xml}')
