# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, Optional, Sequence, Union

import coremltools as ct
import torch

from mmdeploy.utils import get_root_logger
from . import ops  # noqa

try:
    # user might need ops from torchvision
    import torchvision  # noqa
except ImportError:
    pass

SUFFIX_MODE_MAP = {'.mlmodel': 'neuralnetwork', '.mlpackage': 'mlprogram'}


def get_model_suffix(convert_to: str) -> str:
    assert convert_to == 'neuralnetwork' or convert_to == 'mlprogram'
    suffix = ''
    if convert_to == 'neuralnetwork':
        suffix = '.mlmodel'
    if convert_to == 'mlprogram':
        suffix = '.mlpackage'
    return suffix


def create_shape(name: str, default_shape: Sequence, min_shape: Sequence,
                 max_shape: Sequence) -> ct.Shape:
    """Create input shape."""
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
                     output_path: str,
                     input_names: Sequence[str],
                     output_names: Sequence[str],
                     input_shapes: Dict[str, Dict],
                     min_shapes: Dict[str, Dict] = None,
                     max_shapes: Dict[str, Dict] = None,
                     compute_precision: str = 'FLOAT32',
                     convert_to: str = None,
                     minimum_deployment_target: Optional[str] = None,
                     skip_model_load: bool = True):
    """Create a coreml engine from torchscript.

    Args:
        torchscript_model (Union[str, torch.jit.RecursiveScriptModule]):
            The torchscript model to be converted.
        output_path (str): The output file.
        input_names (Sequence[str]): The input names of the model.
        output_names (Sequence[str]): The output names of the model.
        input_shapes (ShapeType): The Default shape of the inputs.
        min_shapes (ShapeType): The minimal shape of the inputs.
        max_shapes (ShapeType): The maximal shape of the inputs.
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

    if min_shapes is None:
        min_shapes = input_shapes
    if max_shapes is None:
        max_shapes = input_shapes
    for name in input_names:
        input_shape = input_shapes[name]
        min_shape = min_shapes.get(name, input_shape)
        max_shape = max_shapes.get(name, input_shape)
        shape = create_shape(name, input_shape, min_shape, max_shape)
        inputs.append(shape)

    for name in output_names:
        outputs.append(ct.TensorType(name=name))

    if convert_to is None:
        suffix = osp.splitext(output_path)[1]
        convert_to = SUFFIX_MODE_MAP[suffix]

    if convert_to not in ['neuralnetwork', 'mlprogram']:
        get_root_logger().warning(f'Unknown postfix: {convert_to}. ',
                                  'Use default mode: neuralnetwork.')

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

    mlmodel.save(output_path)
