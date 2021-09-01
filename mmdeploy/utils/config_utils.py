from typing import Union

import mmcv

from .constants import Backend, Codebase, Task


def load_config(*args):
    """Load the configuration and check the validity."""

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


def get_task_type(deploy_cfg: Union[str, mmcv.Config], default=None) -> Task:
    """Get the task type of the algorithm."""

    deploy_cfg = load_config(deploy_cfg)[0]
    task_pairs = {i.value: i for i in Task}
    task = task_pairs.get(deploy_cfg.get('task', default), default)
    return task


def get_codebase(deploy_cfg: Union[str, mmcv.Config],
                 default=None) -> Codebase:
    """Get the codebase of the config."""

    deploy_cfg = load_config(deploy_cfg)[0]
    codebase_pairs = {i.value: i for i in Codebase}
    codebase = codebase_pairs.get(deploy_cfg.get('codebase', default), default)
    return codebase


def get_backend(deploy_cfg: Union[str, mmcv.Config], default=None) -> Backend:
    """Get the backend of the config."""

    deploy_cfg = load_config(deploy_cfg)[0]
    backend_pairs = {i.value: i for i in Backend}
    backend = backend_pairs.get(deploy_cfg.get('backend', default), default)
    return backend


def is_dynamic_batch(deploy_cfg: Union[str, mmcv.Config],
                     input_name: str = 'input'):
    """Check if input batch is dynamic."""

    deploy_cfg = load_config(deploy_cfg)[0]
    # check if dynamic axes exist
    dynamic_axes = deploy_cfg['pytorch2onnx'].get('dynamic_axes', None)
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
                     input_name: str = 'input'):
    """Check if input shape is dynamic."""

    deploy_cfg = load_config(deploy_cfg)[0]
    # check if dynamic axes exist
    dynamic_axes = deploy_cfg['pytorch2onnx'].get('dynamic_axes', None)
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
