# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from mmdeploy.codebase import BaseTask, get_codebase_class, import_codebase
from mmdeploy.utils import get_codebase, get_task_type


def build_task_processor(model_cfg: mmcv.Config, deploy_cfg: mmcv.Config,
                         device: str) -> BaseTask:
    """Build a task processor to manage the deployment pipeline.

    Args:
        model_cfg (str | mmcv.Config): Model config file.
        deploy_cfg (str | mmcv.Config): Deployment config file.
        device (str): A string specifying device type.

    Returns:
        BaseTask: A task processor.
    """
    codebase_type = get_codebase(deploy_cfg)
    import_codebase(codebase_type)
    codebase = get_codebase_class(codebase_type)
    return codebase.build_task_processor(model_cfg, deploy_cfg, device)


def get_predefined_partition_cfg(deploy_cfg: mmcv.Config, partition_type: str):
    """Get the predefined partition config.

    Notes:
        Currently only support mmdet codebase.

    Args:
        deploy_cfg (mmcv.Config): use deploy config to get the codebase and
            task type.
        partition_type (str): A string specifying partition type.

    Returns:
        dict: A dictionary of partition config.
    """
    codebase_type = get_codebase(deploy_cfg)
    import_codebase(codebase_type)
    task = get_task_type(deploy_cfg)
    codebase = get_codebase_class(codebase_type)
    task_processor_class = codebase.get_task_class(task)
    return task_processor_class.get_partition_cfg(partition_type)
