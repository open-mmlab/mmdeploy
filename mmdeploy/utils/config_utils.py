# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import mmengine

from .constants import Backend, Codebase, Task
from .utils import deprecate, get_root_logger


def load_config(*args) -> List[mmengine.Config]:
    """Load the configuration and check the validity.

    Args:
        args (str | Sequence[str]): The path to the config file(s).

    Returns:
        List[mmengine.Config | dict]: The content of config.
    """

    def _load_config(cfg):
        if isinstance(cfg, str):
            cfg = mmengine.Config.fromfile(cfg)
        if not isinstance(cfg, (dict, mmengine.Config)):
            raise TypeError('deploy_cfg must be a filename or Config object, '
                            f'but got {type(cfg)}')
        return cfg

    assert len(args) > 0
    configs = [_load_config(cfg) for cfg in args]

    return configs


def get_codebase_config(deploy_cfg: Union[str, mmengine.Config]) -> Dict:
    """Get the codebase_config from the config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        Dict : codebase config dict.
    """
    deploy_cfg = load_config(deploy_cfg)[0]
    codebase_config = deploy_cfg.get('codebase_config', {})
    return codebase_config


def get_task_type(deploy_cfg: Union[str, mmengine.Config]) -> Task:
    """Get the task type of the algorithm.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        Task : An enumeration denotes the task type.
    """

    codebase_config = get_codebase_config(deploy_cfg)
    assert 'task' in codebase_config, 'The codebase config of deploy config'\
        'requires a "task" field'
    task = codebase_config['task']
    return Task.get(task)


def register_codebase(codebase: str) -> Codebase:
    """Register a new codebase which is not included in Codebase.

    Args:
        codebase (str): The codebase name.

    Returns:
        Codebase : An enumeration denotes the codebase type.
    """
    from aenum import extend_enum
    try:
        Codebase.get(codebase)
    except Exception as e:
        logger = get_root_logger()
        extend_enum(Codebase, codebase.upper(), codebase)
        logger.warn(f'Failed to get codebase, got: {e}. Then export '
                    f'a new codebase in Codebase {codebase.upper()}: '
                    f'{codebase}')
    return Codebase.get(codebase)


def get_codebase(deploy_cfg: Union[str, mmengine.Config],
                 model_cfg=None) -> Codebase:
    """Get the codebase from the config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        Codebase : An enumeration denotes the codebase type.
    """
    if model_cfg is not None:
        # using mmrazor codebase if the model is a mmrazor model.
        model_cfg: dict = model_cfg['model']
        if model_cfg.get('_scope_', None) == 'mmrazor'\
                or model_cfg['type'].startswith('mmrazor.'):
            return register_codebase('mmrazor')
    codebase_config = get_codebase_config(deploy_cfg)
    assert 'type' in codebase_config, 'The codebase config of deploy config'\
        'requires a "type" field'
    codebase = codebase_config['type']
    return register_codebase(codebase)


def get_backend_config(deploy_cfg: Union[str, mmengine.Config]) -> Dict:
    """Get the backend_config from the config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        Dict : backend config dict.
    """
    deploy_cfg = load_config(deploy_cfg)[0]
    backend_config = deploy_cfg.get('backend_config', {})
    return backend_config


def get_backend(deploy_cfg: Union[str, mmengine.Config]) -> Backend:
    """Get the backend from the config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        Backend: An enumeration denotes the backend type.
    """
    backend_config = get_backend_config(deploy_cfg)
    assert 'type' in backend_config, 'The backend config of deploy config'\
        'requires a "type" field'
    backend = backend_config['type']
    return Backend.get(backend)


def get_ir_config(deploy_cfg: Union[str, mmengine.Config]) -> Dict:
    """Get the IR parameters in export() from config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        Dict: The config dictionary of IR parameters
    """

    deploy_cfg = load_config(deploy_cfg)[0]
    ir_config = deploy_cfg.get('ir_config', None)
    if ir_config is None:
        # TODO: deprecate in future
        ir_config = deploy_cfg.get('onnx_config', {})
    return ir_config


@deprecate(dst_obj=get_ir_config)
def get_onnx_config(deploy_cfg: Union[str, mmengine.Config]) -> Dict:
    """Get the onnx parameters in export() from config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        Dict: The config dictionary of onnx parameters
    """
    onnx_config = get_ir_config(deploy_cfg=deploy_cfg)
    ir_type = onnx_config.get('type', None)
    assert ir_type is None or ir_type == 'onnx', 'Expect IR type is ONNX,'\
        f'but get {ir_type}'
    return onnx_config


def is_dynamic_batch(deploy_cfg: Union[str, mmengine.Config],
                     input_name: Optional[str] = None) -> bool:
    """Check if input batch is dynamic.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.
        input_name (Optional[str]): The name of input in onnx export parameter.

    Returns:
        bool: Is config set dynamic batch (axis 0).
    """

    deploy_cfg = load_config(deploy_cfg)[0]
    ir_config = get_ir_config(deploy_cfg)

    # check if input name is in the config
    input_names = ir_config.get('input_names', None)
    if input_name is None:
        input_name = input_names[0] if input_names else 'input'

    # check if dynamic axes exist
    # TODO: update this when we have more IR
    dynamic_axes = get_dynamic_axes(deploy_cfg)
    if dynamic_axes is None:
        return False

    # check if given input name exist
    input_axes = dynamic_axes.get(input_name, None)
    if input_axes is None:
        return False

    # check if 0 (batch) in input axes
    if 0 in input_axes:
        return True

    return False


def is_dynamic_shape(deploy_cfg: Union[str, mmengine.Config],
                     input_name: Optional[str] = None) -> bool:
    """Check if input shape is dynamic.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.
        input_name (Optional[str]): The name of input in onnx export parameter.

    Returns:
        bool: Is config set dynamic shape (axis 2 and 3).
    """

    # Always dynamic for exporting torchscript
    if get_backend(deploy_cfg) == Backend.TORCHSCRIPT:
        return True

    deploy_cfg = load_config(deploy_cfg)[0]
    ir_config = get_ir_config(deploy_cfg)

    # check if input name is in the config
    input_names = ir_config.get('input_names', None)
    if input_name is None:
        input_name = input_names[0] if input_names else 'input'

    # check if dynamic axes exist
    # TODO: update this when we have more IR
    dynamic_axes = get_dynamic_axes(deploy_cfg)
    if dynamic_axes is None:
        return False

    # check if given input name exist
    input_axes = dynamic_axes.get(input_name, None)
    if input_axes is None:
        return False

    # check if 2 (height) and 3 (width) in input axes
    if 2 in input_axes and 3 in input_axes:
        return True

    return False


def get_input_shape(deploy_cfg: Union[str, mmengine.Config]) -> List[int]:
    """Get the input shape for static exporting.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        List[int]: The input shape for backend model (axis 2 and 3),
            e.g [512, 512].
    """
    input_shape = get_ir_config(deploy_cfg).get('input_shape', None)
    if input_shape is not None:
        assert len(input_shape) == 2, 'length of input_shape should equal to 2'
    return input_shape


def cfg_apply_marks(deploy_cfg: Union[str, mmengine.Config]) -> Optional[bool]:
    """Check if the model needs to be partitioned by checking if the config
    contains 'apply_marks'.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        bool or None: Whether config contains 'apply_marks'.
    """
    partition_config = deploy_cfg.get('partition_config', None)
    if partition_config is None:
        return None

    apply_marks = partition_config.get('apply_marks', False)
    return apply_marks


def get_partition_config(
        deploy_cfg: Union[str, mmengine.Config]) -> Optional[Dict]:
    """Check if the model needs to be partitioned and get the config of
    partition.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        dict or None: The config of partition.
    """
    partition_config = deploy_cfg.get('partition_config', None)
    if partition_config is None:
        return None

    apply_marks = partition_config.get('apply_marks', False)
    if not apply_marks:
        return None

    return partition_config


def get_calib_config(deploy_cfg: Union[str, mmengine.Config]) -> Dict:
    """Check if the model has calibration configs.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        dict: The config of calibration.
    """

    calib_config = deploy_cfg.get('calib_config', None)
    return calib_config


def get_calib_filename(
        deploy_cfg: Union[str, mmengine.Config]) -> Optional[str]:
    """Check if the model needs to create calib and get filename of calib.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        str | None: Could be the filename of output calib file or None.
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


def get_common_config(deploy_cfg: Union[str, mmengine.Config]) -> Dict:
    """Get common parameters from config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        dict: A dict of common parameters for a model.
    """
    backend_config = deploy_cfg['backend_config']
    model_params = backend_config.get('common_config', dict())
    return model_params


def get_quantization_config(deploy_cfg: Union[str, mmengine.Config]) -> Dict:
    """Get quantization parameters from config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        dict: A dict of quantization parameters for a model.
    """
    backend_config = deploy_cfg['backend_config']
    model_params = backend_config.get('quantization_config', dict())
    return model_params


def get_model_inputs(deploy_cfg: Union[str, mmengine.Config]) -> List[Dict]:
    """Get model input parameters from config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        list[dict]: A list of dict containing input parameters for a model.
    """
    backend_config = deploy_cfg['backend_config']
    model_params = backend_config.get('model_inputs', [])
    return model_params


def get_dynamic_axes(
    deploy_cfg: Union[str, mmengine.Config],
    axes_names: List[str] = None
) -> Dict[str, Union[List[int], Dict[int, str]]]:
    """Get model dynamic axes from config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.
        axes_names (List[str]): List with names for dynamic axes.

    Returns:
        Dict[str, Union[List[int], Dict[int, str]]]:
            Dictionary with dynamic axes.
    """
    deploy_cfg = load_config(deploy_cfg)[0]
    ir_config = get_ir_config(deploy_cfg)

    # TODO onnx will be deprecated in the future
    onnx_config = deploy_cfg.get('onnx_config', None)
    if onnx_config is None and ir_config == {}:
        raise KeyError(
            'Field \'onnx_config\' was not found in \'deploy_cfg\'.')
    dynamic_axes = ir_config.get('dynamic_axes', None)
    if dynamic_axes and not isinstance(dynamic_axes, Dict):
        if axes_names is None:
            axes_names = []
            input_names = ir_config.get('input_names', None)
            if input_names:
                axes_names += input_names
            output_names = ir_config.get('output_names', None)
            if output_names:
                axes_names += output_names
            if not axes_names:
                raise KeyError('No names were found to define dynamic axes.')
        dynamic_axes = dict(zip(axes_names, dynamic_axes))
    return dynamic_axes


def get_precision(deploy_cfg: Union[str, mmengine.Config]) -> str:
    """Get precision of config.

    Args:
        deploy_cfg (str | mmengine.Config): The path or content of config.

    Returns:
        str: The precision of target backend.
    """
    precision = 'FP32'
    deploy_cfg = load_config(deploy_cfg)[0]
    backend = get_backend(deploy_cfg=deploy_cfg)
    if backend == Backend.TENSORRT:
        common_cfg = get_common_config(deploy_cfg)
        if common_cfg.get('fp16_mode', False):
            precision = 'FP16'
        if common_cfg.get('int8_mode', False):
            precision = 'INT8'
    if backend == Backend.NCNN and 'precision' in deploy_cfg['backend_config']:
        precision = deploy_cfg['backend_config']['precision']
    return precision


def get_codebase_external_module(
        deploy_cfg: Union[str, mmengine.Config]) -> List:
    return get_codebase_config(deploy_cfg).get('module', [])


def get_normalization(model_cfg: Union[str, mmengine.Config]):
    """Get the Normalize transform from model config.

    Args:
        model_cfg (mmengine.Config): The content of config.
    Returns:
        dict: The Normalize transform.
    """
    model_cfg = load_config(model_cfg)[0]
    data_preprocessor = model_cfg.get('data_preprocessor', {})
    data_preprocessor.update(model_cfg.model.get('data_preprocessor', {}))
    return data_preprocessor


def get_rknn_quantization(deploy_cfg: mmengine.Config):
    """Get the flag of `do_quantization` for rknn backend.

    Args:
        deploy_cfg (mmengine.Config): The content of config.
    Returns:
        bool: Do quantization or not.
    """
    if get_backend(deploy_cfg) == Backend.RKNN:
        return get_backend_config(
            deploy_cfg)['quantization_config']['do_quantization']
    return False
