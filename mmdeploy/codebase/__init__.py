# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from mmengine import Config

from mmdeploy.utils import Codebase
from .base import BaseTask, MMCodebase, get_codebase_class

extra_dependent_library = {
    Codebase.MMOCR: ['mmdet'],
    Codebase.MMROTATE: ['mmdet']
}


def import_codebase(codebase: Codebase, deploy_cfg: Config = None):
    """Import a codebase package in `mmdeploy.codebase`

    The function will check if all dependent libraries are installed.
    For example, to import `mmdeploy.codebase.mmdet`, `mmdet` must be
    installed. To import `mmdeploy.codebase.mmocr`, `mmdet` and `mmocr`
    must be installed.

    Args:
        codebase (Codebase): The codebase to import.
        deploy_cfg (Config): The config of deployment to get external
            modules and dependent libraries.
    """
    codebase_name = codebase.value
    dependent_library = [codebase_name] + \
        extra_dependent_library.get(codebase, [])
    external_module_list = []
    if deploy_cfg:
        external_module_list += deploy_cfg['codebase_config'].get('module', [])
        dependent_library += deploy_cfg['codebase_config']. \
            get('extra_dependent_library', [])
    for lib in dependent_library:
        if (lib == deploy_cfg['codebase_config']['type']
                and len(external_module_list) > 0):
            for external_module in external_module_list:
                importlib.import_module(f'{external_module}')
        else:
            if not importlib.util.find_spec(lib):
                raise ImportError(
                    f'{lib} has not been installed. '
                    f'Import mmdeploy.codebase.{codebase_name} failed.')

            importlib.import_module(f'mmdeploy.codebase.{lib}')
            importlib.import_module(f'{lib}.models')
            importlib.import_module(f'{lib}.datasets')
            importlib.import_module(f'{lib}.structures')
            importlib.import_module(f'{lib}.visualization')
            importlib.import_module(f'{lib}.engine')


__all__ = ['MMCodebase', 'BaseTask', 'get_codebase_class']
