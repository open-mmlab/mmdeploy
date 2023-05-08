# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from subprocess import call
from typing import List, Optional, Union

import onnx

from .init_plugins import get_onnx2ncnn_path


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def get_output_model_file(onnx_path: str,
                          work_dir: Optional[str] = None) -> List[str]:
    """Returns the path to the .param, .bin file with export result.

    Args:
        onnx_path (str): The path of the onnx model.
        work_dir (str|None): The path of the directory for saving the results.
            Defaults to `None`, which means using the directory of onnx_path.

    Returns:
        List[str]: The path of the files where the export result will be
            located.
    """
    if work_dir is None:
        work_dir = osp.dirname(onnx_path)
    mkdir_or_exist(osp.abspath(work_dir))
    file_name = osp.splitext(osp.split(onnx_path)[1])[0]
    save_param = osp.join(work_dir, file_name + '.param')
    save_bin = osp.join(work_dir, file_name + '.bin')
    return [save_param, save_bin]


def from_onnx(onnx_model: Union[onnx.ModelProto, str],
              output_file_prefix: str):
    """Convert ONNX to ncnn.

    The inputs of ncnn include a model file and a weight file. We need to use
    an executable program to convert the `.onnx` file to a `.param` file and
    a `.bin` file. The output files will save to work_dir.

    Example:
        >>> from mmdeploy.apis.ncnn import from_onnx
        >>> onnx_path = 'work_dir/end2end.onnx'
        >>> output_file_prefix = 'work_dir/end2end'
        >>> from_onnx(onnx_path, output_file_prefix)

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
    ret_code = call([onnx2ncnn_path, onnx_path, save_param, save_bin])
    assert ret_code == 0, 'onnx2ncnn failed'
