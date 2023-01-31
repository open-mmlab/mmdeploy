# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, Optional, Sequence, Union

import coremltools as ct
import torch

from mmdeploy.utils import get_root_logger

try:
    # user might need ops from torchvision
    import torchvision  # noqa
except ImportError:
    pass


def get_model_suffix(convert_to: str) -> str:
    assert convert_to == 'neuralnetwork' or convert_to == 'mlprogram'
    suffix = ''
    if convert_to == 'neuralnetwork':
        suffix = '.mlmodel'
    if convert_to == 'mlprogram':
        suffix = '.mlpackage'
    return suffix


def create_shape(name: str, input_shapes: Dict) -> ct.Shape:
    """Create input shape."""
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


def from_torchscript(torchscript_model: Union[str,
                                              torch.jit.RecursiveScriptModule],
                     output_file_prefix: str,
                     input_names: Sequence[str],
                     output_names: Sequence[str],
                     input_shapes: Dict[str, Dict],
                     compute_precision: str = 'FLOAT32',
                     convert_to: str = 'neuralnetwork',
                     minimum_deployment_target: Optional[str] = None,
                     skip_model_load: bool = True,
                     **kwargs):
    """Create a coreml engine from torchscript.

    Args:
        torchscript_model (Union[str, torch.jit.RecursiveScriptModule]):
            The torchscript model to be converted.
        output_file_prefix (str): The output file prefix.
        input_names (Sequence[str]): The input names of the model.
        output_names (Sequence[str]): The output names of the model.
        input_shapes (Dict): The input shapes include max_shape, min_shape and
            default_shape
        compute_precision (str): The model precision,
            FLOAT16 or FLOAT32, see coremltools.precision, default `FLOAT32`.
        convert_to (str): The converted model type, can be
            'neuralnetwork' or 'mlprogram'. Defaults to 'neuralnetwork'.
        minimum_deployment_target (str, optional): minimum deploy target.
            iOS15, iOS16, etc., see coremltools.target
        skip_model_load (bool, optional): Skip model load. Defaults to True.
    """

    try:
        from mmdeploy.backend.torchscript import get_ops_path
        torch.ops.load_library(get_ops_path())
    except Exception as e:
        get_root_logger().warning(
            'Can not load custom ops because:\n'
            f'{e}\n'
            'Some model might not be able to be converted.')

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
        compute_precision = ct.precision[compute_precision]

    mlmodel = ct.convert(
        model=torchscript_model,
        inputs=inputs,
        outputs=outputs,
        compute_precision=compute_precision,
        convert_to=convert_to,
        minimum_deployment_target=ct.target[minimum_deployment_target]
        if minimum_deployment_target else None,
        skip_model_load=skip_model_load)

    suffix = get_model_suffix(convert_to)
    output_path = output_file_prefix + suffix

    mlmodel.save(output_path)
