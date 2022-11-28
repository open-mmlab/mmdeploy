# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from subprocess import call
from typing import List, Optional, Union

import onnx

from .init_plugins import get_onnx2dlc_path


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def get_env_key() -> str:
    """Return environment key str.

    Returns:
        str: The string to find SNPE service URI
    """
    return '__MMDEPLOY_SNPE_URI'


def get_output_model_file(onnx_path: str,
                          work_dir: Optional[str] = None) -> List[str]:
    """Returns the path to the .dlc file with export result.

    Args:
        onnx_path (str): The path to the onnx model.
        work_dir (str|None): The path to the directory for saving the results.
            Defaults to `None`, which means use the directory of onnx_path.

    Returns:
        List[str]: The path to the files where the export result will be
            located.
    """
    if work_dir is None:
        work_dir = osp.dirname(onnx_path)
    mkdir_or_exist(osp.abspath(work_dir))
    file_name = osp.splitext(osp.split(onnx_path)[1])[0]
    save_dlc = osp.join(work_dir, file_name + '.dlc')
    return save_dlc


def from_onnx(onnx_model: Union[onnx.ModelProto, str],
              output_file_prefix: str):
    """Convert ONNX to dlc.

    We need to use a executable program to convert the `.onnx` file to a `.dlc`

    Example:
        >>> from mmdeploy.apis.snpe import from_onnx
        >>> onnx_path = 'work_dir/end2end.onnx'
        >>> output_file_prefix = 'work_dir/end2end'
        >>> from_onnx(onnx_path, output_file_prefix)

    Args:
        onnx_path (ModelProto|str): The path of the onnx model.
        output_file_prefix (str): The path to save the output .dlc file.
    """

    if not isinstance(onnx_model, str):
        onnx_path = tempfile.NamedTemporaryFile(suffix='.onnx').name
        onnx.save(onnx_model, onnx_path)
    else:
        onnx_path = onnx_model

    save_dlc = output_file_prefix + '.dlc'

    onnx2dlc = get_onnx2dlc_path()
    ret_code = call(
        [onnx2dlc, '--input_network', onnx_path, '--output', save_dlc])
    assert ret_code == 0, 'onnx2dlc failed'
