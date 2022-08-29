# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmdeploy.utils.device import parse_cuda_device_id
from .utils import create_runtime, register_engines


def from_onnx(onnx_model: str,
              output_file_prefix: str,
              device: str = 'cuda:0',
              input_shapes: Optional[Sequence[Sequence[int]]] = None,
              **kwargs):
    """Convert ONNX to PPLNN.

    PPLNN is capable of optimizing onnx model. The optimized algorithm is saved
    into `algo_file` in json format. Note that `input_shapes` actually require
    multiple shapes of inputs in its original design. But in the pipeline of
    our codebase, we only pass one input shape which can be modified by users'
    own preferences.

    Args:
        output_file_prefix (str): File path to save PPLNN optimization
            algorithm and ONNX file
        onnx_model (str): Input onnx model.
        device (str): A string specifying device, defaults to 'cuda:0'.
        input_shapes (Sequence[Sequence[int]] | None): Shapes for PPLNN
            optimization, default to None.

    Examples:
        >>> from mmdeploy.apis.pplnn import from_onnx
        >>>
        >>> from_onnx(onnx_model = 'example.onnx',
                      output_file_prefix = 'example')
    """
    if device == 'cpu':
        device_id = -1
    else:
        assert 'cuda' in device, f'unexpected device: {device}, must contain '
        '`cpu` or `cuda`'
        device_id = parse_cuda_device_id(device)
    if input_shapes is None:
        input_shapes = [[1, 3, 224,
                         224]]  # PPLNN default shape for optimization

    algo_file = output_file_prefix + '.json'
    onnx_output_path = output_file_prefix + '.onnx'
    engines = register_engines(
        device_id,
        disable_avx512=False,
        quick_select=False,
        export_algo_file=algo_file,
        input_shapes=input_shapes)
    _ = create_runtime(onnx_model, engines)  # side effect: export algorithms
    import shutil
    if onnx_output_path != onnx_model:
        shutil.copy2(onnx_model, onnx_output_path)
