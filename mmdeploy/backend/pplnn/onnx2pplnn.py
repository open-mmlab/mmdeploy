# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence

import onnx

from mmdeploy.utils.device import parse_cuda_device_id
from .utils import create_runtime, register_engines


def from_onnx(onnx_file: str,
              output_file: str,
              algo_file: Optional[str] = None,
              device: str = 'cuda:0',
              input_shapes: Optional[Dict[str, Sequence[int]]] = None,
              disable_avx512: bool = False,
              quick_select: bool = False,
              **kwargs):
    """Convert ONNX to PPLNN.
    PPLNN is capable of optimizing onnx model. The optimized algorithm is saved
    into `algo_file` in json format. Note that `input_shapes` actually require
    multiple shapes of inputs in its original design. But in the pipeline of
    our codebase, we only pass one input shape which can be modified by users'
    own preferences.
    Args:
        onnx_file (str): Input onnx model.
        output_file (str): Path of output ONNX model file.
        algo_file (str): Path of PPLNN algorithm file.
        device (str): A string specifying device, defaults to 'cuda:0'.
        input_shapes (Dict[str, Sequence[int]] | None): Shapes for PPLNN
            optimization, default to None.
        disable_avx512 (bool): Whether to disable avx512 for x86.
            Defaults to `False`.
        quick_select (bool): Whether to use default algorithms.
            Defaults to `False`.
    Examples:
        >>> from mmdeploy.apis.pplnn import from_onnx
        >>>
        >>> from_onnx(onnx_file = 'example.onnx',
                      output_file_prefix = 'example')
    """
    if device == 'cpu':
        device_id = -1
    else:
        assert 'cuda' in device, f'unexpected device: {device}, must contain '
        '`cpu` or `cuda`'
        device_id = parse_cuda_device_id(device)

    onnx_model = onnx.load(onnx_file)
    input_names = [i.name for i in onnx_model.graph.input]

    if input_shapes is None:
        input_shapes = [[1, 3, 224,
                         224]]  # PPLNN default shape for optimization
    elif isinstance(input_shapes, Dict):
        input_shapes = [input_shapes[name] for name in input_names]

    engines = register_engines(
        device_id,
        disable_avx512=disable_avx512,
        quick_select=quick_select,
        export_algo_file=algo_file,
        input_shapes=input_shapes)
    _ = create_runtime(onnx_file, engines)  # side effect: export algorithms
    import shutil
    if output_file != onnx_file:
        shutil.copy2(onnx_file, output_file)
