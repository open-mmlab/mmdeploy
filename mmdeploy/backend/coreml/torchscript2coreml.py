# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Union

import coremltools as ct
import mmcv
import torch

from mmdeploy.utils import (get_common_config, get_model_inputs,
                            get_root_logger, load_config)
from mmdeploy.utils.config_utils import get_ir_config

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


def from_torchscript(model_id: int,
                     torchscript_model: Union[str,
                                              torch.jit.RecursiveScriptModule],
                     output_file_prefix: str, deploy_cfg: Union[str,
                                                                mmcv.Config],
                     backend_files: List[str], **kwargs):
    """Create a coreml engine from torchscript.

    Args:
         model_id (int): Index of input model.
        torchscript_model (Union[str, torch.jit.RecursiveScriptModule]):
            The torchscript model to be converted.
        output_file_prefix (str): The output file prefix.
        deploy_cfg (str | mmcv.Config): Deployment config.
        backend_files (List[str]):
            Backend files used by deployment for testing pipeline
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

    deploy_cfg = load_config(deploy_cfg)[0]

    common_params = get_common_config(deploy_cfg)
    model_params = get_model_inputs(deploy_cfg)[model_id]

    final_params = common_params
    final_params.update(model_params)

    ir_config = get_ir_config(deploy_cfg)

    input_names = ir_config.get('input_names', [])
    input_shapes = final_params['input_shapes']
    inputs = []

    for name in input_names:
        shape = create_shape(name, input_shapes[name])
        inputs.append(shape)

    output_names = ir_config.get('output_names', [])
    outputs = []

    for name in output_names:
        outputs.append(ct.TensorType(name=name))

    convert_to = deploy_cfg.backend_config.convert_to
    if convert_to == 'neuralnetwork':
        # Compute precision must be None for neuralnetwork conversion
        compute_precision = None
    else:
        compute_precision = ct.precision[final_params.get(
            'compute_precision', 'FLOAT32')]

    minimum_deployment_target = final_params.get('minimum_deployment_target',
                                                 None)

    mlmodel = ct.convert(
        model=torchscript_model,
        inputs=inputs,
        outputs=outputs,
        compute_precision=compute_precision,
        convert_to=convert_to,
        minimum_deployment_target=ct.target[minimum_deployment_target]
        if minimum_deployment_target else None,
        skip_model_load=final_params.get('skip_model_load', False))

    suffix = get_model_suffix(convert_to)
    output_path = output_file_prefix + suffix
    backend_files.append(output_path)
    mlmodel.save(output_path)
