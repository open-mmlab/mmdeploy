# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import mmcv

from .constants import Backend, Codebase, Task


def load_config(*args) -> List[mmcv.Config]:
    """Load the configuration and check the validity.

    Args:
        args (str | Sequence[str]): The path to the config file(s).

    Returns:
        List[mmcv.Config]: The content of config.
    """

    def _load_config(cfg):
        if isinstance(cfg, str):
            cfg = mmcv.Config.fromfile(cfg)
        if not isinstance(cfg, (mmcv.Config, mmcv.ConfigDict)):
            raise TypeError('deploy_cfg must be a filename or Config object, '
                            f'but got {type(cfg)}')
        return cfg

    assert len(args) > 0
    configs = [_load_config(cfg) for cfg in args]

    return configs


def get_codebase_config(deploy_cfg: Union[str, mmcv.Config]) -> Dict:
    """Get the codebase_config from the config.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        Dict : codebase config dict.
    """
    deploy_cfg = load_config(deploy_cfg)[0]
    codebase_config = deploy_cfg.get('codebase_config', {})
    return codebase_config


def get_task_type(deploy_cfg: Union[str, mmcv.Config]) -> Task:
    """Get the task type of the algorithm.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        Task : An enumeration denotes the task type.
    """

    codebase_config = get_codebase_config(deploy_cfg)
    assert 'task' in codebase_config, 'The codebase config of deploy config'\
        'requires a "task" field'
    task = codebase_config['task']
    return Task.get(task)


def get_codebase(deploy_cfg: Union[str, mmcv.Config]) -> Codebase:
    """Get the codebase from the config.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        Codebase : An enumeration denotes the codebase type.
    """

    codebase_config = get_codebase_config(deploy_cfg)
    assert 'type' in codebase_config, 'The codebase config of deploy config'\
        'requires a "type" field'
    codebase = codebase_config['type']
    return Codebase.get(codebase)


def get_backend_config(deploy_cfg: Union[str, mmcv.Config]) -> Dict:
    """Get the backend_config from the config.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        Dict : backend config dict.
    """
    deploy_cfg = load_config(deploy_cfg)[0]
    backend_config = deploy_cfg.get('backend_config', {})
    return backend_config


def get_backend(deploy_cfg: Union[str, mmcv.Config]) -> Backend:
    """Get the backend from the config.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        Backend: An enumeration denotes the backend type.
    """
    backend_config = get_backend_config(deploy_cfg)
    assert 'type' in backend_config, 'The backend config of deploy config'\
        'requires a "type" field'
    backend = backend_config['type']
    return Backend.get(backend)


def get_onnx_config(deploy_cfg: Union[str, mmcv.Config]) -> Dict:
    """Get the onnx parameters in export() from config.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        Dict: The config dictionary of onnx parameters
    """

    deploy_cfg = load_config(deploy_cfg)[0]
    onnx_config = deploy_cfg.get('onnx_config', {})
    return onnx_config


def is_dynamic_batch(deploy_cfg: Union[str, mmcv.Config],
                     input_name: str = 'input') -> bool:
    """Check if input batch is dynamic.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.
        input_name (str): The name of input in onnx export parameter.

    Returns:
        bool: Is config set dynamic batch (axis 0).
    """

    deploy_cfg = load_config(deploy_cfg)[0]
    # check if dynamic axes exist
    dynamic_axes = get_onnx_config(deploy_cfg).get('dynamic_axes', None)
    if dynamic_axes is None:
        return False

    # check if given input name exist
    input_axes = dynamic_axes.get(input_name, None)
    if input_axes is None:
        return False

    # check if 2 and 3 in input axes
    if 0 in input_axes:
        return True

    return False


def is_dynamic_shape(deploy_cfg: Union[str, mmcv.Config],
                     input_name: Optional[str] = None) -> bool:
    """Check if input shape is dynamic.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.
        input_name (str): The name of input in onnx export parameter.

    Returns:
        bool: Is config set dynamic shape (axis 2 and 3).
    """

    deploy_cfg = load_config(deploy_cfg)[0]

    # check if input name is in the config
    if input_name is None:
        input_names = deploy_cfg.onnx_config.get('input_names', None)
        input_name = input_names[0] if input_names else 'input'

    # check if dynamic axes exist
    dynamic_axes = get_onnx_config(deploy_cfg).get('dynamic_axes', None)
    if dynamic_axes is None:
        return False

    # check if given input name exist
    input_axes = dynamic_axes.get(input_name, None)
    if input_axes is None:
        return False

    # check if 2 and 3 in input axes
    if 2 in input_axes and 3 in input_axes:
        return True

    return False


def get_input_shape(deploy_cfg: Union[str, mmcv.Config]) -> List[int]:
    """Get the input shape for static exporting.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        List[int]: The input shape for backend model (axis 2 and 3),
            e.g [512, 512].
    """
    input_shape = get_onnx_config(deploy_cfg)['input_shape']
    if input_shape is not None:
        assert len(input_shape) == 2, 'length of input_shape should equal to 2'
    return input_shape


def cfg_apply_marks(deploy_cfg: Union[str, mmcv.Config]) -> Union[bool, None]:
    """Check if the model needs to be partitioned by checking if the config
    contains 'apply_marks'.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        bool or None: Whether config contains 'apply_marks'.
    """
    partition_config = deploy_cfg.get('partition_config', None)
    if partition_config is None:
        return None

    apply_marks = partition_config.get('apply_marks', False)
    return apply_marks


def get_partition_config(deploy_cfg: Union[str, mmcv.Config]) -> Dict:
    """Check if the model needs to be partitioned and get the config of
    partition.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        dict: The config of partition.
    """
    partition_config = deploy_cfg.get('partition_config', None)
    if partition_config is None:
        return None

    apply_marks = partition_config.get('apply_marks', False)
    if not apply_marks:
        return None

    return partition_config


def get_calib_config(deploy_cfg: Union[str, mmcv.Config]) -> Dict:
    """Check if the model has calibration configs.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        dict: The config of calibration.
    """

    calib_config = deploy_cfg.get('calib_config', None)
    return calib_config


def get_calib_filename(deploy_cfg: Union[str, mmcv.Config]) -> str:
    """Check if the model needs to create calib and get filename of calib.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        str: The filename of output calib file.
    """

    calib_config = get_calib_config(deploy_cfg)
    if calib_config is None:
        return None
    create_calib = calib_config.get('create_calib', False)
    if create_calib:
        calib_filename = calib_config.get('calib_file', 'calib_file.h5')
        return calib_filename
    else:
        return None


def get_common_config(deploy_cfg: Union[str, mmcv.Config]) -> Dict:
    """Get common parameters from config.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        dict: A dict of common parameters for a model.
    """
    backend_config = deploy_cfg['backend_config']
    model_params = backend_config.get('common_config', dict())
    return model_params


def get_model_inputs(deploy_cfg: Union[str, mmcv.Config]) -> List[Dict]:
    """Get model input parameters from config.

    Args:
        deploy_cfg (str | mmcv.Config): The path or content of config.

    Returns:
        list[dict]: A list of dict containing input parameters for a model.
    """
    backend_config = deploy_cfg['backend_config']
    model_params = backend_config.get('model_inputs', [])
    return model_params
