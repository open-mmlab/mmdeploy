# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import mmcv

from mmdeploy.backend.openvino import ModelOptimizerOptions
from mmdeploy.utils import get_model_inputs
from mmdeploy.utils.config_utils import get_backend_config, get_ir_config


def update_input_names(input_info: Dict[str, List],
                       input_names: List[str]) -> Dict[str, List]:
    """Replaces the default input name in 'input_info' with the value from the
    deployment config, if they differ.

    Args:
        input_info (Dict[str, List]): Names and shapes of input.
        input_names (List[str]): Input names from the deployment config.

    Returns:
        Dict[str, List]: A dict that stores the names and shapes of input.
    """
    input_info_keys = set(input_info.keys())
    input_names = set(input_names)
    if input_info_keys != input_names:
        old_names = input_info_keys - input_names
        new_names = input_names - input_info_keys
        for new_key, old_key in zip(new_names, old_names):
            input_info[new_key] = input_info.pop(old_key)
    return input_info


def get_input_info_from_cfg(deploy_cfg: mmcv.Config) -> Dict[str, List]:
    """Get the input names and shapes from the configs for OpenVINO Model
    Optimizer.

    Args:
        deploy_cfg (mmcv.Config): Deployment config.

    Returns:
        Dict[str, List]: A dict that stores the names and shapes of input.
    """
    # The partition is not supported now. Set the id of model to 0.
    model_inputs = get_model_inputs(deploy_cfg)[0]
    input_info = model_inputs['opt_shapes']
    ir_config = get_ir_config(deploy_cfg)
    if ir_config is not None:
        input_names = ir_config.get('input_names', None)
        if input_names:
            if not isinstance(input_info, Dict):
                input_info = dict(zip(input_names, input_info))
            input_info = update_input_names(input_info, input_names)
    return input_info


def get_mo_options_from_cfg(deploy_cfg: mmcv.Config) -> ModelOptimizerOptions:
    """Get additional parameters for the Model Optimizer from the deploy
    config.

    Args:
        deploy_cfg (mmcv.Config): Deployment config.

    Returns:
        ModelOptimizerOptions: A class that will contain additional arguments.
    """
    backend_config = get_backend_config(deploy_cfg)
    mo_options = backend_config.get('mo_options', None)
    mo_options = ModelOptimizerOptions(mo_options)
    return mo_options
