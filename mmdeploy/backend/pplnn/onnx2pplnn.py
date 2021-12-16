# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from pyppl import nn as pplnn

from mmdeploy.utils.device import parse_cuda_device_id
from .wrapper import register_engines


def onnx2pplnn(algo_file: str,
               onnx_model: str,
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
        algo_file (str): File path to save PPLNN optimization algorithm.
        onnx_model (str): Input onnx model.
        device (str): A string specifying device, defaults to 'cuda:0'.
        input_shapes (Sequence[Sequence[int]] | None): Shapes for PPLNN
            optimization, default to None.

    Examples:
        >>> from mmdeploy.apis.pplnn import onnx2pplnn
        >>>
        >>> onnx2pplnn(algo_file = 'example.json', onnx_model = 'example.onnx')
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

    engines = register_engines(
        device_id,
        disable_avx512=False,
        quick_select=False,
        export_algo_file=algo_file,
        input_shapes=input_shapes)
    runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(
        onnx_model, engines)
    assert runtime_builder is not None, 'Failed to create '\
        'OnnxRuntimeBuilder.'
