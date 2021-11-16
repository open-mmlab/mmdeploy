from typing import Optional, Sequence

import torch
from pyppl import nn as pplnn

from mmdeploy.apis.ppl import register_engines


def parse_cuda_device_id(device: str) -> int:
    """Parse cuda device index from a string.

    Args:
        device (str): The typical style of string specifying cuda device,
            e.g.: 'cuda:0'.

    Returns:
        int: The parsed device id, defaults to `0`.
    """
    device_id = 0
    if len(device) >= 6:
        device_id = torch.device(device).index
    return device_id


def onnx2ppl(algo_file: str,
             onnx_model: str,
             device: str = 'cuda:0',
             input_shapes: Optional[Sequence[Sequence[int]]] = None,
             **kwargs):
    """Convert ONNX to PPL.

    PPL is capable of optimizing onnx model. The optimized algorithm is saved
    into `algo_file` in json format. Note that `input_shapes` actually require
    multiple shapes of inputs in its original design. But in the pipeline of
    our codebase, we only pass one input shape which can be modified by users'
    own preferences.

    Args:
        algo_file (str): File path to save PPL optimization algorithm.
        onnx_model (str): Input onnx model.
        device (str): A string specifying cuda device, defaults to 'cuda:0'.
        input_shapes (Sequence[Sequence[int]] | None): shapes for PPL
            optimization, default to None.

    Examples:
        >>> from mmdeploy.apis.ppl import onnx2ppl
        >>>
        >>> onnx2ppl(algo_file = 'example.json', onnx_model = 'example.onnx')
    """
    if device == 'cpu':
        device_id = -1
    else:
        assert 'cuda' in device, f'unexpected device: {device}, must contain '
        '`cpu` or `cuda`'
        device_id = parse_cuda_device_id(device)
    if input_shapes is None:
        input_shapes = [[1, 3, 224, 224]]  # PPL default shape for optimization

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
