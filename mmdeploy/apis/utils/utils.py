# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Any, Optional, Sequence

import mmengine

from mmdeploy.codebase import BaseTask, get_codebase_class, import_codebase
from mmdeploy.utils import (get_backend, get_codebase, get_task_type,
                            parse_device_id)
from mmdeploy.utils.config_utils import get_codebase_external_module
from ..core import PIPELINE_MANAGER


def check_backend_device(deploy_cfg: mmengine.Config, device: str):
    """Check if device is appropriate for the backend.

    Args:
        deploy_cfg (str | mmengine.Config): Deployment config file.
        device (str): A string specifying device type.
    """
    backend = get_backend(deploy_cfg).value
    device_id = parse_device_id(device)
    mismatch = dict(
        tensorrt=lambda id: id == -1,
        openvino=lambda id: id > -1,
    )
    if backend in mismatch and mismatch[backend](device_id):
        raise ValueError(f'{device} is invalid for the backend {backend}')


def build_task_processor(model_cfg: mmengine.Config,
                         deploy_cfg: mmengine.Config, device: str) -> BaseTask:
    """Build a task processor to manage the deployment pipeline.

    Args:
        model_cfg (str | mmengine.Config): Model config file.
        deploy_cfg (str | mmengine.Config): Deployment config file.
        device (str): A string specifying device type.

    Returns:
        BaseTask: A task processor.
    """
    check_backend_device(deploy_cfg=deploy_cfg, device=device)
    codebase_type = get_codebase(deploy_cfg, model_cfg=model_cfg)
    custom_module_list = get_codebase_external_module(deploy_cfg)
    import_codebase(codebase_type, custom_module_list)
    codebase = get_codebase_class(codebase_type)
    return codebase.build_task_processor(model_cfg, deploy_cfg, device)


def get_predefined_partition_cfg(deploy_cfg: mmengine.Config,
                                 partition_type: str):
    """Get the predefined partition config.

    Notes:
        Currently only support mmdet codebase.

    Args:
        deploy_cfg (mmengine.Config): use deploy config to get the codebase and
            task type.
        partition_type (str): A string specifying partition type.

    Returns:
        dict: A dictionary of partition config.
    """
    codebase_type = get_codebase(deploy_cfg)
    custom_module_list = get_codebase_external_module(deploy_cfg)
    import_codebase(codebase_type, custom_module_list)
    task = get_task_type(deploy_cfg)
    codebase = get_codebase_class(codebase_type)
    task_processor_class = codebase.get_task_class(task)
    return task_processor_class.get_partition_cfg(partition_type)


@PIPELINE_MANAGER.register_pipeline()
def to_backend(backend_name: str,
               ir_files: Sequence[str],
               work_dir: str,
               deploy_cfg: Optional[Any] = None,
               log_level: int = logging.INFO,
               device: str = 'cpu',
               **kwargs) -> Sequence[str]:
    """Convert intermediate representation to given backend.

    Args:
        backend_name (str): The name of the backend.
        ir_files (Sequence[str]): The intermediate representation files.
        work_dir (str): The work directory, backend files and logs should
            be save in this directory.
        deploy_cfg (Any): The deploy config.
        log_level (int, optional): The log level. Defaults to logging.INFO.
        device (str, optional): The device type. Defaults to 'cpu'.
    Returns:
        Sequence[str]: Backend files.
    """
    from mmdeploy.backend.base import get_backend_manager
    backend_mgr = get_backend_manager(backend_name)
    return backend_mgr.to_backend(
        ir_files=ir_files,
        work_dir=work_dir,
        deploy_cfg=deploy_cfg,
        log_level=log_level,
        device=device,
        **kwargs)
