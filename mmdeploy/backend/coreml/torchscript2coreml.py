# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Optional, Union, Sequence, Dict
import coremltools as ct
import torch


def get_model_suffix(convert_to: str) -> str:
    assert convert_to == 'neuralnetwork' or convert_to == 'mlprogram'
    suffix = ''
    if convert_to == 'neuralnetwork':
        suffix = '.mlmodel'
    if convert_to == 'mlprogram':
        suffix = '.mlpackage'
    return suffix


def create_shape(name: str,
                 input_shapes: Dict) -> ct.Shape:
    """Create input shape
    """
    min_shape = input_shapes['min_shape']
    max_shape = input_shapes['max_shape']
    default_shape = input_shapes['default_shape']
    assert len(min_shape) == len(max_shape) == len(default_shape)
    shape = []
    n_dim = len(min_shape)
    for i in range(n_dim):
        low = min_shape[i]
        high = max_shape[i]
        assert low <= high
        if low == -1 or high == -1:
            shape.append(ct.RangeDim())
        elif low == high:
            shape.append(low)
        else:
            shape.append(ct.RangeDim(low, high))

    shape = ct.Shape(shape=shape, default=default_shape)
    return ct.TensorType(shape=shape, name=name)


def from_torchscript(torchscript_model: Union[str, torch.jit.RecursiveScriptModule],
                     output_file_prefix: str,
                     input_names: Sequence[str],
                     output_names: Sequence[str],
                     input_shapes: Dict,
                     convert_to: str = 'neuralnetwork',
                     fp16_mode: bool = False,
                     skip_model_load: bool = True,
                     **kwargs):
    """Create a coreml engine from torchscript.
    """

    if isinstance(torchscript_model, str):
        torchscript_model = torch.jit.load(torchscript_model)

    inputs = []
    outputs = []

    for name in input_names:
        shape = create_shape(name, input_shapes[name])
        inputs.append(shape)

    for name in output_names:
        outputs.append(ct.TensorType(name=name))

    if convert_to == 'neuralnetwork':
        compute_precision = None
    else:
        if fp16_mode:
            compute_precision = ct.precision.FLOAT16
        else:
            compute_precision = ct.precision.FLOAT32

    mlmodel = ct.convert(model=torchscript_model,
                         inputs=inputs, outputs=outputs,
                         compute_precision=compute_precision,
                         convert_to=convert_to,
                         skip_model_load=False)

    suffix = get_model_suffix(convert_to)
    output_path = output_file_prefix + suffix
    mlmodel.save(output_path)
