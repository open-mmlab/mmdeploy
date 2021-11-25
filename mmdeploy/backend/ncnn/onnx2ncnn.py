from subprocess import call
from typing import List

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
    save_param = onnx_path.replace('.onnx', '.param')
    save_bin = onnx_path.replace('.onnx', '.bin')

    return [save_param, save_bin]


def onnx2ncnn(onnx_path: str, work_dir: str):
    """Convert ONNX to ncnn.

    The inputs of ncnn include a model file and a weight file. We need to use
    a executable program to convert the ".onnx" file to a ".param" file and
    a ".bin" file. The output files will save to work_dir.

    Args:
        onnx_path (str): The path of the onnx model.
        work_dir (str): The path to the directory for saving the results.
    """

    onnx2ncnn_path = get_onnx2ncnn_path()

    save_param, save_bin = get_output_model_file(onnx_path, work_dir)

    call([onnx2ncnn_path, onnx_path, save_param, save_bin])
