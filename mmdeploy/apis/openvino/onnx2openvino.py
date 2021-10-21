import os.path as osp
import subprocess
from subprocess import CalledProcessError, run
from typing import Dict, List, Union

import torch


def is_mo_available() -> bool:
    """Checking if OpenVINO Model Optimizer is available.

    Returns:
        bool: True, if Model Optimizer is available, else - False.
    """
    is_available = True
    try:
        run('mo.py -h',
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
            check=True)
    except CalledProcessError:
        is_available = False
    return is_available


def get_output_model_file(onnx_path: str, work_dir: str) -> str:
    """Returns the path to the .xml file with export result.

    Args:
        onnx_path (str): The path to the onnx model.
        work_dir (str): The path to the directory for saving the results.

    Returns:
        str: The path to the file where the export result will be located.
    """
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

    if not is_mo_available():
        raise RuntimeError(
            'OpenVINO Model Optimizer is not found or configured improperly')

    mo_args = f'--input_model="{onnx_path}" '\
              f'--output_dir="{work_dir}" ' \
              f'--output="{output}" ' \
              f'--input="{input_names}" ' \
              f'--input_shape="{input_shapes}" ' \
              f'--disable_fusing '
    command = f'mo.py {mo_args}'
    print(f'Args for mo.py: {command}')
    mo_output = run(command, capture_output=True, shell=True, check=True)
    print(mo_output.stdout.decode())
    print(mo_output.stderr.decode())

    model_xml = get_output_model_file(onnx_path, work_dir)
    print(f'Successfully exported OpenVINO model: {model_xml}')
