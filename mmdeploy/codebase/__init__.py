# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmdeploy.utils import Codebase
from .base import BaseTask, MMCodebase, get_codebase_class

extra_dependent_library = {
    Codebase.MMOCR: ['mmdet'],
    Codebase.MMROTATE: ['mmdet']
}


def import_codebase(codebase_type: Codebase, custom_module_list: List = []):
    """Import a codebase package in `mmdeploy.codebase`

    The function will check if all dependent libraries are installed.
    For example, to import `mmdeploy.codebase.mmdet`, `mmdet` must be
    installed. To import `mmdeploy.codebase.mmocr`, `mmdet` and `mmocr`
    must be installed.

    Args:
        codebase (Codebase): The codebase to import.
    """
    import importlib
    codebase_name = codebase_type.value

    dependent_library = [codebase_name] + \
        extra_dependent_library.get(codebase_type, [])
    for lib in dependent_library + custom_module_list:
        if not importlib.util.find_spec(lib):
            raise ImportError(f'{lib} has not been installed. '
                              f'Import {lib} failed.')
    if len(custom_module_list) > 0:
        for custom_module in custom_module_list:
            importlib.import_module(f'{custom_module}')
    codebase = get_codebase_class(codebase_type)
    codebase.register_all_modules()


__all__ = ['MMCodebase', 'BaseTask', 'get_codebase_class']
