# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmcv


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
