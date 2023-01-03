# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdeploy
from mmdeploy.utils import get_codebase_version, get_root_logger


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDeploy'] = f'{mmdeploy.__version__}+{get_git_hash()[:7]}'

    return env_info


def check_backend():
    from mmdeploy.backend.base import get_backend_manager
    from mmdeploy.utils import Backend
    exclude_backend_lists = [Backend.DEFAULT, Backend.PYTORCH, Backend.SDK]
    backend_lists = [
        backend for backend in Backend if backend not in exclude_backend_lists
    ]

    for backend in backend_lists:
        backend_mgr = get_backend_manager(backend.value)
        backend_mgr.check_env(logger.info)


def check_codebase():
    codebase_versions = get_codebase_version()
    for k, v in codebase_versions.items():
        logger.info(f'{k}:\t{v}')


if __name__ == '__main__':
    logger = get_root_logger()
    logger.info('\n')
    logger.info('**********Environmental information**********')
    for name, val in collect_env().items():
        logger.info('{}: {}'.format(name, val))
    logger.info('\n')
    logger.info('**********Backend information**********')
    check_backend()
    logger.info('\n')
    logger.info('**********Codebase information**********')
    check_codebase()
